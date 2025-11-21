from django.db.models import Count
from resume_matcher.models import SavedJob

duplicates = SavedJob.objects.values('user', 'job_title', 'company_name') \
    .annotate(count=Count('id')) \
    .filter(count__gt=1)

for duplicate in duplicates:
    user = duplicate['user']
    job_title = duplicate['job_title']
    company_name = duplicate['company_name']

    # Get all instances of the duplicate job for this user
    duplicate_jobs = SavedJob.objects.filter(
        user=user,
        job_title=job_title,
        company_name=company_name
    ).order_by('created_at') # Keep the oldest one

    # Delete all but the first (oldest) instance
    for job_to_delete in duplicate_jobs[1:]:
        job_to_delete.delete()

print("Duplicate SavedJob entries removed.")
