from django.urls import path

from text_recognition import views

urlpatterns = [
    path('', views.InitialView.as_view(), name='initial'),
    path('censar/', views.IndexView.as_view(), name='index'),
    path('censar/result/', views.ResultView.as_view(), name='result'),
    path('censar/images/', views.ImagesView.as_view(), name='images'),
]
