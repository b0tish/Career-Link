from .views import RegisterAPI, LoginAPI, UserAPI, AdminCreateJobProviderAPI
from django.urls import path
from knox import views as knox_views

urlpatterns = [
    path('api/auth/register', RegisterAPI.as_view(), name='register'),
    path('api/auth/login', LoginAPI.as_view(), name='login'),
    path('api/auth/user', UserAPI.as_view(), name='user'),
    path('api/auth/logout', knox_views.LogoutView.as_view(), name='logout'),
    path('api/auth/admin/create_job_provider', AdminCreateJobProviderAPI.as_view(), name='create_job_provider'),
]
