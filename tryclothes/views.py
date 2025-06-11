from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import BodyScan , User
import os
from django.conf import settings
from django.contrib.auth import authenticate, login ,logout
from django.contrib import messages
import base64
from django.views.decorators.csrf import csrf_exempt






# Create your views here.
from django.http import JsonResponse
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import tensorflow as tf
from tensorflow.keras.utils import get_file
import tensorflow_hub as hub
import uuid

# utils.py
import cv2
import numpy as np
import requests
from PIL import Image
import io
from rembg import remove
import json
from sklearn.cluster import KMeans
import mediapipe as mp



# Model URL
# model_url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'

# # Download and untar the model to a directory
# model_dir = get_file('ssd_mobilenet_v2_coco', model_url, untar=True)

# # Print the directory to confirm
# print(f"Model directory: {model_dir}")

# # Set the path to the saved model directory inside the downloaded folder
# model_path = os.path.join(model_dir, 'saved_model')

# # Check if the path exists
# if not os.path.exists(model_path):
#     print(f"Model path does not exist: {model_path}")
# else:
#     print(f"Model path exists: {model_path}")

# # Load the model
# model = tf.saved_model.load(model_path)

model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")




# def get_clothing_images_amazon(keyword="tshirt"):
#     headers = {'User-Agent': 'Mozilla/5.0'}
#     url = f"https://www.amazon.in/s?k={keyword}"
#     r = requests.get(url, headers=headers)
#     soup = BeautifulSoup(r.content, "html.parser")

#     images = []
#     for img_tag in soup.select("img.s-image"):
#         images.append(img_tag['src'])
#     return images

# Scraping function to get clothing product image from Amazon
def get_clothing_images_amazon(product_url):
    """Enhanced Amazon scraper with better image detection"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(product_url)
        time.sleep(5)  # Wait for page to load
        
        # Try multiple selectors for Amazon product images
        image_selectors = [
            '#landingImage',
            '#imgBlkFront',
            '.a-dynamic-image',
            '[data-a-image-name="landingImage"]',
            '.a-spacing-small img',
            '#main-image-container img',
            '.imgTagWrapper img',
            '[data-old-hires]',
            '[data-a-dynamic-image]'
        ]
        
        image_url = None
        for selector in image_selectors:
            try:
                image_element = driver.find_element(By.CSS_SELECTOR, selector)
                # Try different attributes
                for attr in ['data-old-hires', 'data-a-hires', 'src', 'data-src']:
                    image_url = image_element.get_attribute(attr)
                    if image_url and 'http' in image_url:
                        # Get high resolution version
                        if '._' in image_url:
                            image_url = image_url.split('._')[0] + '._AC_SL1500_.jpg'
                        break
                if image_url:
                    break
            except:
                continue
                
        return image_url
        
    except Exception as e:
        print(f"Error scraping Amazon: {e}")
        return None
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
        image_url = get_clothing_images_amazon(product_link)
        print(f"Scraped image URL: {image_url}")
        
        if not image_url:
            return JsonResponse({
                'success': False,
                'error': 'Could not find product image. Please check the URL and try again.'
            }, status=400)
        
        try:
            # Extract clothing from the image
            # Replace this with your actual clothing extraction function
            result = extract_clothing_from_image(image_url)
            print(f"Clothing extraction result: {result.get('success', False)}")
            
            if result.get('success'):
                return JsonResponse({
                    'success': True,
                    'original_image': image_url,
                    'extracted_clothing': result['clothing_image'],
                    'message': result.get('message', 'Clothing extracted successfully')
                })
            else:
                return JsonResponse({
                    'success': False,
                    'original_image': image_url,
                    'error': result.get('error', 'Failed to extract clothing from image')
                }, status=422)
                
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return JsonResponse({
                'success': False,
                'original_image': image_url,
                'error': f'Failed to process image: {str(e)}'
            }, status=500)
    
    except Exception as e:
        print(f"General error in scrape_product_image: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)
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




def extract_clothing_from_image(image_url):
    """
    Extract any clothing item from a model image using computer vision techniques
    Works for: shirts, t-shirts, jeans, pants, dresses, pajamas, jackets, etc.
    
    Args:
        image_url (str): URL of the product image
        
    Returns:
        dict: JSON response with extracted clothing image or error
    """
    try:
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(response.content))
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Method 1: Background removal using rembg (works for all clothing)
        clothing_image = extract_with_background_removal(image, opencv_image)
        
        # Method 2: If background removal fails, try person segmentation
        if clothing_image is None:
            clothing_image = extract_with_person_segmentation(opencv_image)
        
        # Method 3: If both fail, try adaptive clothing detection
        if clothing_image is None:
            clothing_image = extract_with_adaptive_detection(opencv_image)
        
        if clothing_image is not None:
            # Convert result to base64 for JSON response
            _, buffer = cv2.imencode('.png', clothing_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'clothing_image': f"data:image/png;base64,{img_base64}",
                'message': 'Clothing extracted successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Could not extract clothing from image',
                'original_image': image_url
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Error processing image: {str(e)}',
            'original_image': image_url
        }
    

def extract_with_background_removal(pil_image, opencv_image):
    """
    Method 1: Use rembg to remove background and isolate any clothing item
    """
    try:
        # Remove background using rembg
        no_bg_image = remove(pil_image)
        
        # Convert back to OpenCV format
        no_bg_cv = cv2.cvtColor(np.array(no_bg_image), cv2.COLOR_RGBA2BGR)
        
        # Create mask for non-transparent areas
        alpha = np.array(no_bg_image)[:, :, 3]
        mask = alpha > 0
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (person with clothing)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create bounding box around the person
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding to ensure we don't cut off clothing
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(opencv_image.shape[1] - x, w + 2*padding)
            h = min(opencv_image.shape[0] - y, h + 2*padding)
            
            # Apply the mask to the original image
            result = opencv_image.copy()
            result[mask == 0] = [255, 255, 255]  # White background
            
            # Crop to the clothing region
            if w > 50 and h > 50:  # Ensure minimum size
                cropped_clothing = result[y:y+h, x:x+w]
                return cropped_clothing
                
    except Exception as e:
        print(f"Background removal method failed: {e}")
        return None

def extract_with_person_segmentation(opencv_image):
    """
    Method 2: Use advanced person segmentation to isolate clothing
    """
    try:
        height, width = opencv_image.shape[:2]
        
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2LAB)
        
        # Create a mask to focus on the center region (where person usually is)
        center_mask = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        
        # Create elliptical region around center
        cv2.ellipse(center_mask, (center_x, center_y), 
                   (width//3, height//2), 0, 0, 360, 255, -1)
        
        # Use GrabCut algorithm for foreground extraction
        mask = np.zeros((height, width), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle around the person (center region)
        rect = (width//4, height//8, width//2, 3*height//4)
        
        # Apply GrabCut
        cv2.grabCut(opencv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        
        # Find contours and get the largest one
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2*padding)
            h = min(height - y, h + 2*padding)
            
            # Apply mask to original image
            result = opencv_image.copy()
            result[mask2 == 0] = [255, 255, 255]
            
            if w > 50 and h > 50:
                cropped_clothing = result[y:y+h, x:x+w]
                return cropped_clothing
                
    except Exception as e:
        print(f"Person segmentation method failed: {e}")
        return None

def extract_with_adaptive_detection(opencv_image):
    """
    Method 3: Adaptive clothing detection using multiple techniques
    """
    try:
        height, width = opencv_image.shape[:2]
        
        # Convert to different color spaces
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection with multiple scales
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Dilate edges to connect clothing regions
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area, aspect ratio, and position
            valid_contours = []
            min_area = (width * height) * 0.02  # At least 2% of image
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # More flexible criteria for different clothing types
                    # Shirts/tops: wider, upper portion
                    # Pants/jeans: taller, lower portion  
                    # Dresses: tall, full body
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Check if contour is in reasonable position and size
                    if (0.3 < aspect_ratio < 3.0 and 
                        center_x > width*0.2 and center_x < width*0.8 and
                        center_y > height*0.1 and center_y < height*0.9):
                        valid_contours.append(contour)
            
            if valid_contours:
                # Get the largest valid contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2*padding)
                h = min(height - y, h + 2*padding)
                
                # Create mask for the clothing region
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                # Apply mask to original image
                result = opencv_image.copy()
                result[mask == 0] = [255, 255, 255]
                
                cropped_clothing = result[y:y+h, x:x+w]
                return cropped_clothing
                
    except Exception as e:
        print(f"Adaptive detection method failed: {e}")
        return None




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