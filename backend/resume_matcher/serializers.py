from rest_framework import serializers
from .models import SavedJob, CVHistory

class SavedJobSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    class Meta:
        model = SavedJob
        fields = '__all__'

class CVHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = CVHistory
        fields = '__all__'
