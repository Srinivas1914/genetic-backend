from django.core.management.base import BaseCommand
from django.utils import timezone
from app.models import ModelPerformance
import pickle
import os

class Command(BaseCommand):
    help = 'Populate ModelPerformance table with trained model results'

    def handle(self, *args, **options):
        # Clear existing data
        ModelPerformance.objects.all().delete()

        # Results from training (from ml_trainer.py output)
        model_results = [
            {'model_name': 'Decision Tree', 'accuracy': 100.00, 'precision': 100.00, 'recall': 100.00, 'f1_score': 100.00, 'training_time': 0.38},
            {'model_name': 'K-Nearest Neighbors', 'accuracy': 98.42, 'precision': 98.56, 'recall': 98.42, 'f1_score': 98.48, 'training_time': 2.00},
            {'model_name': 'Random Forest', 'accuracy': 92.65, 'precision': 92.84, 'recall': 92.65, 'f1_score': 92.74, 'training_time': 12.99},
            {'model_name': 'SVM', 'accuracy': 85.12, 'precision': 85.34, 'recall': 85.12, 'f1_score': 85.23, 'training_time': 136.73},
        ]

        # Create ModelPerformance entries
        for result in model_results:
            ModelPerformance.objects.create(
                model_name=result['model_name'],
                accuracy=result['accuracy'],
                precision=result['precision'],
                recall=result['recall'],
                f1_score=result['f1_score'],
                training_time=result['training_time'],
                last_trained=timezone.now()
            )

        self.stdout.write(self.style.SUCCESS(f'Successfully populated {len(model_results)} model performances'))