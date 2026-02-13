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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
            'Support Vector Machine': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42),
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
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'predictions': y_pred
            }
            
            print(f"✓ {name} completed")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Training Time: {training_time:.2f}s")
        
    def build_neural_network(self, input_dim):
        """Build standard Neural Network"""
        model = Sequential([
            Dense(128, input_dim=input_dim),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            Dense(64),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            Dense(32),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.2),
            
            Dense(16),
            Activation('relu'),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_cnn_model(self, input_dim):
        """Build Convolutional Neural Network"""
        model = Sequential([
            Input(shape=(input_dim, 1)),
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_rnn_model(self, input_dim):
        """Build Recurrent Neural Network with LSTM"""
        model = Sequential([
            Input(shape=(input_dim, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_deep_learning_models(self):
        """Train all deep learning models"""
        print("\n" + "="*60)
        print("Training Deep Learning Models")
        print("="*60)
        
        input_dim = self.X_train.shape[1]
        
        # Reshape data for CNN and RNN
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        dl_models = {
            'Neural Network': (self.build_neural_network(input_dim), self.X_train, self.X_test),
            'CNN': (self.build_cnn_model(input_dim), X_train_reshaped, X_test_reshaped),
            'RNN-LSTM': (self.build_rnn_model(input_dim), X_train_reshaped, X_test_reshaped),
        }
        
        for name, (model, X_tr, X_te) in dl_models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            history = model.fit(
                X_tr, self.y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Predictions
            y_pred_prob = model.predict(X_te, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred) * 100
            precision = precision_score(self.y_test, y_pred, average='weighted') * 100
            recall = recall_score(self.y_test, y_pred, average='weighted') * 100
            f1 = f1_score(self.y_test, y_pred, average='weighted') * 100
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'predictions': y_pred,
                'history': history.history
            }
            
            print(f"✓ {name} completed")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Training Time: {training_time:.2f}s")
    
    def train_all_models(self):
        """Train all models"""
        self.load_and_preprocess_data()
        self.train_traditional_ml_models()
        self.train_deep_learning_models()
        return self.results
    
    def save_models(self, save_dir='app/ml_models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scaler
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save traditional ML models
        for name, model in self.models.items():
            if name not in ['Neural Network', 'CNN', 'RNN-LSTM']:
                with open(f'{save_dir}/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # Save deep learning models
        if 'Neural Network' in self.models:
            self.models['Neural Network'].save(f'{save_dir}/neural_network.h5')
        if 'CNN' in self.models:
            self.models['CNN'].save(f'{save_dir}/cnn.h5')
        if 'RNN-LSTM' in self.models:
            self.models['RNN-LSTM'].save(f'{save_dir}/rnn_lstm.h5')
        
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
