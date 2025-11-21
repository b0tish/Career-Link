from django.urls import path
from .views import ResumeUploadView, JobProviderDashboardView, SavedJobView, DashboardView, JobView

urlpatterns = [
    path('upload/', ResumeUploadView.as_view(), name='upload_resume'),
    path('job-provider-dashboard/', JobProviderDashboardView.as_view(), name='job_provider_dashboard'),
    path('save-job/', SavedJobView.as_view(), name='save_job'),
    path('user-dashboard/', DashboardView.as_view(), name='user_dashboard'),
    path('saved-jobs/<int:pk>/', SavedJobView.as_view(), name='saved_job_detail'),
    path('cv-history/<int:pk>/', ResumeUploadView.as_view(), name='cv_history_detail'),
    path('jobs/<str:job_id>/', JobView.as_view(), name='job_detail'),
    path('jobs/', JobView.as_view(), name='jobs'),
]

