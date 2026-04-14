from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    profile_image = models.ImageField(upload_to='profiles/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"

class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=50)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'User Activities'
    
    def __str__(self):
        return f"{self.user.username} - {self.activity_type} at {self.timestamp}"

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    gene_1 = models.FloatField()
    gene_2 = models.FloatField()
    gene_3 = models.FloatField()
    gene_4 = models.FloatField()
    gene_5 = models.FloatField()
    gene_6 = models.FloatField()
    gene_7 = models.FloatField()
    gene_8 = models.FloatField()
    gene_9 = models.FloatField()
    gene_10 = models.FloatField()
    snp_1 = models.IntegerField()
    snp_2 = models.IntegerField()
    snp_3 = models.IntegerField()
    snp_4 = models.IntegerField()
    snp_5 = models.IntegerField()
    age = models.IntegerField()
    bmi = models.FloatField()
    smoking_status = models.IntegerField()
    protein_similarity = models.FloatField()
    
    predicted_result = models.IntegerField()
    prediction_confidence = models.FloatField()
    model_used = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Prediction Histories'
    
    def __str__(self):
        return f"Prediction by {self.user.username} at {self.created_at}"

class ModelPerformance(models.Model):
    model_name = models.CharField(max_length=100, unique=True)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    training_time = models.FloatField()
    last_trained = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-accuracy']
    
    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.2f}%"
