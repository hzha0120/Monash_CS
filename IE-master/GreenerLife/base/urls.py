from django.conf.urls import url
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index, name = "home"),
    path('e_waste/', views.e_waste, name = "e_waste"),
    path('clothing/', views.clothing, name = "clothing"),
    path('index/', views.index, name = "index"),
    path('about_us/', views.about_us, name = "about_us"),
    path('e_waste_classification/', views.e_waste_classification, name = "e_waste_classification"),
    path('garbage/', views.garbage, name = "garbage"),
    path('garbageClassification/', views.garbageClassification, name = "garbageClassification"),
    path('mask_feed', views.video, name='video'),
    path('garbage_video', views.garbage_video, name='garbage_video'),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

