from django.core.management.base import BaseCommand
from app.ml_trainer import GeneticDisorderPredictor
from app.models import ModelPerformance
import os

class Command(BaseCommand):
    help = 'Train all ML/DL models and save performance metrics'

    def handle(self, *args, **options):
        data_path = 'genetic_data.csv'
        
        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(f'Data file not found: {data_path}'))
            return
        
        self.stdout.write(self.style.SUCCESS('Starting model training...'))
        
        # Train models
        predictor = GeneticDisorderPredictor(data_path)
        results = predictor.train_all_models()
        predictor.save_models()
        
        # Save to database
        for model_name, metrics in results.items():
            ModelPerformance.objects.update_or_create(
                model_name=model_name,
                defaults={
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'training_time': metrics['training_time']
                }
            )
        
        self.stdout.write(self.style.SUCCESS('✓ All models trained successfully!'))
        self.stdout.write(self.style.SUCCESS(f'✓ {len(results)} models saved'))
