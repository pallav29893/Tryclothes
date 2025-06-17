from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import BodyScan , User
import os
from django.conf import settings
from django.contrib.auth import authenticate, login ,logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import tensorflow_hub as hub
import uuid
import cv2
import numpy as np
from PIL import Image
import json
import mediapipe as mp

import torch
import open3d as o3d
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import shutil

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")

def load_midas_model():
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.eval()

    transform = Compose([
        Resize(384),  # Input size for DPT_Large
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return midas, transform

def estimate_depth(image_path, midas, transform):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)

    depth_map = prediction.squeeze().cpu().numpy()
    return depth_map


def depth_to_point_cloud(depth_map, image_path, scale=1.0):
    # Use OpenCV to read image as NumPy array (not PIL!)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    h, w = depth_map.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    x = (xx - w / 2) * depth_map / scale
    y = (yy - h / 2) * depth_map / scale
    z = depth_map

    points = np.stack((x, -y, z), axis=-1).reshape(-1, 3)
    colors = img.reshape(-1, 3)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc



def reconstruct_mesh_from_images(image_paths):
    midas, transform = load_midas_model()
    all_pcs = []

    for img_path in image_paths:
        print(f"[INFO] Processing image: {img_path}")
        depth = estimate_depth(img_path, midas, transform)
        pc = depth_to_point_cloud(depth, img_path)
        all_pcs.append(pc)

    # Combine point clouds
    combined_pc = all_pcs[0]
    for pc in all_pcs[1:]:
        combined_pc += pc

    combined_pc = combined_pc.voxel_down_sample(voxel_size=2.0)
    combined_pc.estimate_normals()

    # Create mesh using ball pivoting
    distances = combined_pc.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        combined_pc, o3d.utility.DoubleVector([radius, radius * 2])
    )

    model_path = f"/tmp/generated_model_{uuid.uuid4().hex}.ply"
    o3d.io.write_triangle_mesh(model_path, mesh)
    return {"model_path": model_path, "message": "3D model created"}



def get_all_clothing_images_amazon(product_url):
    """Scrape all product images from an Amazon product page."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 ...')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    image_urls = []

    try:
        driver.get(product_url)
        time.sleep(5)

        # Find all thumbnail images in the image block
        thumbnails = driver.find_elements(By.CSS_SELECTOR, '.imageThumbnail .a-button-thumbnail img')

        for thumb in thumbnails:
            try:
                thumb_url = thumb.get_attribute('src')

                # Convert thumbnail to high-res
                if thumb_url and '._' in thumb_url:
                    high_res = thumb_url.split('._')[0] + '._AC_SL1500_.jpg'
                else:
                    high_res = thumb_url

                if high_res and high_res not in image_urls:
                    image_urls.append(high_res)
            except Exception as e:
                print("Thumbnail parse error:", e)

        # Fallback to main image if no thumbnails found
        if not image_urls:
            main_image = driver.find_element(By.ID, 'landingImage')
            main_url = main_image.get_attribute('src')
            image_urls.append(main_url)

        return image_urls

    except Exception as e:
        print(f"Error scraping images: {e}")
        return []

    finally:
        driver.quit()


@csrf_exempt
def scrape_product_image(request):
    """Handle product image scraping and clothing extraction"""
    
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Handle both form data and JSON data
        if request.content_type == 'application/json':
            try:
                body = request.body.decode('utf-8')
                print("Raw body:", body)
                data = json.loads(body)
                product_link = data.get('product_link')
            except json.JSONDecodeError as e:
                return JsonResponse({'success': False, 'error': f'Invalid JSON: {str(e)}'}, status=400)
        else:
            product_link = request.POST.get('product_link')
        
        if not product_link:
            return JsonResponse({
                'success': False,
                'error': 'Product link is required'
            }, status=400)
        
        print(f"Processing product URL: {product_link}")
        
        # Scrape the product image
        image_urls = get_all_clothing_images_amazon(product_link)
        print(f"Found {len(image_urls)} images")
        print(image_urls,"imageeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

        if not image_urls:
            return JsonResponse({'success': False, 'error': 'No images found'})

        # Process images to 3D
        try:
            model_result = build_3d_model_from_images(image_urls)
            print("Model result:", model_result)
            # Move GLB to a media-accessible path


            glb_path = model_result['model_path']
            filename = os.path.basename(glb_path)
            public_dir = os.path.join(settings.MEDIA_ROOT, 'models')
            os.makedirs(public_dir, exist_ok=True)

            public_path = os.path.join(public_dir, filename)
            shutil.copy(glb_path, public_path)

            # URL to be used in frontend
            model_url = f"/media/models/{filename}"
            print(model_url,"modellllllurllllll")

            return JsonResponse({
                'success': True,
                'original_image': image_urls[0],
                'model_path': model_url,  # now a URL
                'message': model_result['message']
            })
        except Exception as e:
            import traceback
            traceback.print_exc()  # for better logs
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
                
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return JsonResponse({
                'success': False,
                'original_image': image_urls,
                'error': f'Failed to process image: {str(e)}'
            }, status=500)
    
    except Exception as e:
        print(f"General error in scrape_product_image: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)
    
# def build_3d_model_from_images(image_urls):
#     try:
#         image_paths = []
#         for url in image_urls:
#             img_data = requests.get(url).content
#             path = f'/tmp/{uuid.uuid4()}.jpg'
#             with open(path, 'wb') as f:
#                 f.write(img_data)
#             image_paths.append(path)

#         result = reconstruct_mesh_from_images(image_paths)
#         return result

#     except Exception as e:
#         print(f"[ERROR in build_3d_model_from_images] {e}")
#         raise

import trimesh


def build_3d_model_from_images(image_urls):
    try:
        image_paths = []
        for url in image_urls:
            img_data = requests.get(url).content
            path = f'/tmp/{uuid.uuid4()}.jpg'
            with open(path, 'wb') as f:
                f.write(img_data)
            image_paths.append(path)

        # Generate the PLY model
        result = reconstruct_mesh_from_images(image_paths)  # Must return a dict with 'model_path'

        ply_path = result.get("model_path")
        if not ply_path or not ply_path.endswith(".ply"):
            raise ValueError("Invalid model path or format")

        # Convert to GLB using trimesh
        mesh = trimesh.load(ply_path)
        glb_filename = os.path.basename(ply_path).replace(".ply", ".glb")
        glb_path = os.path.join("/tmp", glb_filename)
        mesh.export(glb_path)

        return {
            "model_path": glb_path,  # This will be the .glb path now
            "message": "3D model (GLB) created"
        }

    except Exception as e:
        print(f"[ERROR in build_3d_model_from_images] {e}")
        raise


# def scrape_product_image(request):
#     if request.method == 'POST':
#         product_link = request.POST.get('product_link')
#         image_url = get_clothing_images_amazon(product_link)
#         print("Scraped image URL:", image_url)

#         if image_url:
#             try:
#                 cropped_image_url = extract_shirt_from_image(image_url)
#                 print(cropped_image_url,"cropeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
#                 if cropped_image_url:
#                     return JsonResponse({'product_image': cropped_image_url})
#                 else:
#                     return JsonResponse({'error': 'Shirt not detected in image'}, status=404)
#             except Exception as e:
#                 print("Image processing error:", e)
#                 return JsonResponse({'error': 'Failed to process image'}, status=500)
#         else:
#             return JsonResponse({'error': 'Product image not found'}, status=400)







def get_body_measurements(image_path):
    mp_pose = mp.solutions.pose
    mp_landmarks = mp.solutions.pose.PoseLandmark  # import the enum

    pose = mp_pose.Pose(static_image_mode=True)
    image = cv2.imread(image_path)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return {"error": "Body not detected"}

    landmarks = results.pose_landmarks.landmark

    top = landmarks[mp_landmarks.NOSE].y
    bottom = landmarks[mp_landmarks.LEFT_ANKLE].y  # use LEFT_ANKLE instead of ANKLE_LEFT
    height_ratio = bottom - top

    image_height = image.shape[0]
    approximate_height_cm = image_height * height_ratio * 0.5  # adjust scale factor

    return {
        "approx_height_cm": round(approximate_height_cm, 2),
        "note": "Estimated based on 2D image; not 100% accurate"
    }

@login_required
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            scan = BodyScan.objects.create(user=request.user, image=image_file)

            # Call ML function to get measurements
            image_path = os.path.join(settings.MEDIA_ROOT, scan.image.name)
            measurements = get_body_measurements(image_path)

            # Optionally save to user or another model
            scan.measurements = measurements
            scan.save()

            return render(request, 'results.html', {'measurements': measurements})

    return render(request, 'upload_image.html')

def signup(request):
    if request.method == 'POST':
        username = request.POST.get('name')
        email = request.POST.get('email')
        number = request.POST.get('number')

        password = request.POST.get('password')

        user = User.objects.create_user(username=username,email=email,phone_number=number,password=password)

        return redirect('login')

    return render(request,'signup.html')

def login_view(request):
    error_message = None
    if request.method == 'POST':
        username = request.POST.get('name')
        password = request.POST.get('password')

        user = authenticate(request=request, username=username, password=password)
        if user:
            login(request,user)
            return redirect('upload_image')
        else:
            error_message = "Invalid Email or Password."

    return render(request,'login.html',{'error_message': error_message})

def logout_user(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')

def profile(request):
    profile = request.user
    return render(request,'profile.html',{'profile':profile})

def profile_edit(request):
    profile = request.user
    if request.method == 'POST':
        username = request.POST.get('name')
        phone_number = request.POST.get('number')
        email = request.POST.get('email')
        city = request.POST.get('city')
        state = request.POST.get('state')
        country = request.POST.get('country')
        image = request.FILES.get('profile_image')  
        print(image,"jnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")

        profile.username = username
        profile.email = email
        profile.phone_number = phone_number
        profile.city = city
        profile.state = state
        profile.country = country
        
        if image:
            profile.user_profile_image = image  

        profile.save()

        messages.success(request, "Profile updated successfully!")
        return redirect('profile') 

    return render(request, 'profile_edit.html', {'profile': profile})