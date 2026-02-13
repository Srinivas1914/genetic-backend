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
            {'model_name': 'Extra Trees', 'accuracy': 99.57, 'precision': 99.58, 'recall': 99.57, 'f1_score': 99.57, 'training_time': 5.06},
            {'model_name': 'Logistic Regression', 'accuracy': 99.49, 'precision': 99.50, 'recall': 99.49, 'f1_score': 99.49, 'training_time': 0.07},
            {'model_name': 'Support Vector Machine', 'accuracy': 99.42, 'precision': 99.42, 'recall': 99.42, 'f1_score': 99.42, 'training_time': 136.73},
            {'model_name': 'Random Forest', 'accuracy': 99.23, 'precision': 99.24, 'recall': 99.23, 'f1_score': 99.23, 'training_time': 12.99},
            {'model_name': 'XGBoost', 'accuracy': 99.21, 'precision': 99.22, 'recall': 99.21, 'f1_score': 99.21, 'training_time': 1.10},
            {'model_name': 'LightGBM', 'accuracy': 99.16, 'precision': 99.17, 'recall': 99.16, 'f1_score': 99.16, 'training_time': 0.80},
            {'model_name': 'RNN-LSTM', 'accuracy': 99.14, 'precision': 99.15, 'recall': 99.14, 'f1_score': 99.14, 'training_time': 831.57},
            {'model_name': 'Neural Network', 'accuracy': 98.70, 'precision': 98.73, 'recall': 98.70, 'f1_score': 98.70, 'training_time': 68.35},
            {'model_name': 'Gradient Boosting', 'accuracy': 98.20, 'precision': 98.26, 'recall': 98.20, 'f1_score': 98.20, 'training_time': 23.96},
            {'model_name': 'Decision Tree', 'accuracy': 98.15, 'precision': 98.19, 'recall': 98.15, 'f1_score': 98.15, 'training_time': 0.38},
            {'model_name': 'AdaBoost', 'accuracy': 97.94, 'precision': 98.00, 'recall': 97.94, 'f1_score': 97.94, 'training_time': 5.33},
            {'model_name': 'Naive Bayes', 'accuracy': 93.32, 'precision': 93.59, 'recall': 93.32, 'f1_score': 93.31, 'training_time': 0.02},
            {'model_name': 'CNN', 'accuracy': 93.28, 'precision': 93.60, 'recall': 93.28, 'f1_score': 93.26, 'training_time': 57.08},
            {'model_name': 'K-Nearest Neighbors', 'accuracy': 90.54, 'precision': 91.92, 'recall': 90.54, 'f1_score': 90.47, 'training_time': 2.00},
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