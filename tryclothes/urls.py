from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # path('',views.get_clothing_images_amazon,name="get_clothing_images_amazon"),
    path('',views.upload_image,name="upload_image"),
    path('scrape_product/', views.scrape_product_image, name='scrape_product'),
    path('signup/',views.signup,name='signup'),
    path('login/',views.login_view,name='login'),
    path('logout/',views.logout_user,name='logout'),
    path('profile/',views.profile,name='profile'),
    path('profile_edit/',views.profile_edit,name='profile_edit'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

