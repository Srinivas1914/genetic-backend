from django.contrib import admin
from .models import UserProfile, UserActivity, PredictionHistory, ModelPerformance

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone']
    list_filter = ['created_at']

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'activity_type', 'timestamp', 'ip_address']
    list_filter = ['activity_type', 'timestamp']
    search_fields = ['user__username', 'description']
    ordering = ['-timestamp']

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'predicted_result', 'prediction_confidence', 'model_used', 'created_at']
    list_filter = ['predicted_result', 'model_used', 'created_at']
    search_fields = ['user__username']
    ordering = ['-created_at']

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'last_trained']
    ordering = ['-accuracy']
