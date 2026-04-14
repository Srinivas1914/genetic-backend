import numpy as np
import pandas as pd
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Traditional ML Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class GeneticDisorderPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the genetic data"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_traditional_ml_models(self):
        """Train all traditional ML models"""
        print("\n" + "="*60)
        print("Training Traditional Machine Learning Models")
        print("="*60)
        
        ml_models = {
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        }
        
        for name, model in ml_models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred) * 100
            precision = precision_score(self.y_test, y_pred, average='weighted') * 100
            recall = recall_score(self.y_test, y_pred, average='weighted') * 100
            f1 = f1_score(self.y_test, y_pred, average='weighted') * 100
            conf_matrix = confusion_matrix(self.y_test, y_pred).tolist()
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'training_time': training_time,
                'predictions': y_pred
            }
            
            print(f"✓ {name} completed")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Training Time: {training_time:.2f}s")
        

    
    def train_all_models(self):
        """Train all models"""
        self.load_and_preprocess_data()
        self.train_traditional_ml_models()
        return self.results
    
    def save_models(self, save_dir='app/ml_models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save traditional ML models
        for name, model in self.models.items():
            with open(f'{save_dir}/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save results
        with open(f'{save_dir}/results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"\n✓ All models saved to {save_dir}/")
    
    def get_comparison_data(self):
        """Get comparison data for visualization"""
        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'training_time': metrics['training_time']
            })
        return sorted(comparison, key=lambda x: x['accuracy'], reverse=True)

def main():
    """Main training function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ml_trainer.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("GENETIC DISORDER PREDICTION - ML/DL TRAINING SYSTEM")
    print("="*60)
    
    predictor = GeneticDisorderPredictor(data_path)
    results = predictor.train_all_models()
    predictor.save_models()
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison = predictor.get_comparison_data()
    
    for i, model_data in enumerate(comparison, 1):
        print(f"\n{i}. {model_data['model']}")
        print(f"   Accuracy:  {model_data['accuracy']:.2f}%")
        print(f"   Precision: {model_data['precision']:.2f}%")
        print(f"   Recall:    {model_data['recall']:.2f}%")
        print(f"   F1-Score:  {model_data['f1_score']:.2f}%")
        print(f"   Time:      {model_data['training_time']:.2f}s")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
