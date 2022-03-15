from django.urls import path
from . import views

urlpatterns = [
    path('', views.Start.as_view()),
    path('index/', views.Index.as_view()),
    path('detect/<int:id>',views.Detect.as_view()),
    path('history/', views.History.as_view()),  
]