from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
import json
import pickle
import numpy as np
import os
import pandas as pd

from .models import UserProfile, UserActivity, PredictionHistory, ModelPerformance

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_activity(user, activity_type, description, request):
    """Log user activity"""
    UserActivity.objects.create(
        user=user,
        activity_type=activity_type,
        description=description,
        ip_address=get_client_ip(request)
    )

def login_view(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username_or_email = request.POST.get('username')
        password = request.POST.get('password')
        
        # Try to authenticate with username first
        user = authenticate(request, username=username_or_email, password=password)
        
        # If authentication failed, try with email
        if user is None:
            try:
                user_obj = User.objects.get(email=username_or_email)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                user = None
        
        if user is not None:
            login(request, user)
            log_activity(user, 'LOGIN', f'User logged in successfully', request)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username/email or password')
    
    return render(request, 'login.html')

def register_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Validation
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
        elif password != confirm_password:
            messages.error(request, 'Passwords do not match')
        elif len(password) < 6:
            messages.error(request, 'Password must be at least 6 characters')
        else:
            # Create user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            
            # Create user profile
            UserProfile.objects.create(user=user)
            
            log_activity(user, 'REGISTRATION', 'New user registered', request)
            messages.success(request, 'Registration successful! Please login.')
            return redirect('login')
    
    return render(request, 'register.html')

@login_required
def logout_view(request):
    """User logout"""
    log_activity(request.user, 'LOGOUT', 'User logged out', request)
    logout(request)
    messages.info(request, 'You have been logged out')
    return redirect('login')

@login_required
def dashboard_view(request):
    """Main dashboard"""
    # Get model performance data
    model_performances = ModelPerformance.objects.all()
    
    # Get user's recent predictions
    recent_predictions = PredictionHistory.objects.filter(user=request.user)[:5]
    
    # Get user activity stats
    total_predictions = PredictionHistory.objects.filter(user=request.user).count()
    
    context = {
        'user': request.user,
        'model_performances': model_performances,
        'recent_predictions': recent_predictions,
        'total_predictions': total_predictions,
    }
    
    return render(request, 'dashboard.html', context)

@login_required
def predict_view(request):
    """Prediction view with all models"""
    if request.method == 'POST':
        try:
            # Get input features
            gene_1 = float(request.POST.get('gene_1'))
            gene_2 = float(request.POST.get('gene_2'))
            gene_3 = float(request.POST.get('gene_3'))
            gene_4 = float(request.POST.get('gene_4'))
            gene_5 = float(request.POST.get('gene_5'))
            gene_6 = float(request.POST.get('gene_6'))
            gene_7 = float(request.POST.get('gene_7'))
            gene_8 = float(request.POST.get('gene_8'))
            gene_9 = float(request.POST.get('gene_9'))
            gene_10 = float(request.POST.get('gene_10'))
            snp_1 = int(request.POST.get('snp_1'))
            snp_2 = int(request.POST.get('snp_2'))
            snp_3 = int(request.POST.get('snp_3'))
            snp_4 = int(request.POST.get('snp_4'))
            snp_5 = int(request.POST.get('snp_5'))
            age = int(request.POST.get('age'))
            bmi = float(request.POST.get('bmi'))
            smoking_status = int(request.POST.get('smoking_status'))
            protein_similarity = float(request.POST.get('protein_similarity'))
            
            # Prepare input array
            input_features = np.array([[
                gene_1, gene_2, gene_3, gene_4, gene_5,
                gene_6, gene_7, gene_8, gene_9, gene_10,
                snp_1, snp_2, snp_3, snp_4, snp_5,
                age, bmi, smoking_status, protein_similarity
            ]])
            
            # Load scaler
            with open('app/ml_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            # Scale input
            input_scaled = scaler.transform(input_features)
            
            # Get predictions from all models
            predictions = {}
            model_files = {
                'K-Nearest Neighbors': 'k-nearest_neighbors.pkl',
                'Random Forest': 'random_forest.pkl',
                'SVM': 'support_vector_machine.pkl',
                'Decision Tree': 'decision_tree.pkl',
                'Logistic Regression': 'logistic_regression.pkl',
                'Gradient Boosting': 'gradient_boosting.pkl',
                'XGBoost': 'xgboost.pkl',
                'LightGBM': 'lightgbm.pkl',
                'Naive Bayes': 'naive_bayes.pkl',
                'AdaBoost': 'adaboost.pkl',
                'Extra Trees': 'extra_trees.pkl',
            }
            
            for model_name, file_name in model_files.items():
                try:
                    with open(f'app/ml_models/{file_name}', 'rb') as f:
                        model = pickle.load(f)
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0]
                    predictions[model_name] = {
                        'prediction': int(pred),
                        'confidence': float(max(prob) * 100)
                    }
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
            
            # Load deep learning models
            from tensorflow import keras
            
            dl_models = {
                'Neural Network': 'neural_network.h5',
                'CNN': 'cnn.h5',
                'RNN-LSTM': 'rnn_lstm.h5',
            }
            
            for model_name, file_name in dl_models.items():
                try:
                    model = keras.models.load_model(f'app/ml_models/{file_name}')
                    
                    if model_name in ['CNN', 'RNN-LSTM']:
                        input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)
                        prob = model.predict(input_reshaped, verbose=0)[0][0]
                    else:
                        prob = model.predict(input_scaled, verbose=0)[0][0]
                    
                    pred = 1 if prob > 0.5 else 0
                    predictions[model_name] = {
                        'prediction': int(pred),
                        'confidence': float(prob * 100 if pred == 1 else (1 - prob) * 100)
                    }
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
            
            # Calculate ensemble prediction
            pred_values = [p['prediction'] for p in predictions.values()]
            ensemble_pred = 1 if sum(pred_values) > len(pred_values) / 2 else 0
            ensemble_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
            
            # Save prediction history
            PredictionHistory.objects.create(
                user=request.user,
                gene_1=gene_1, gene_2=gene_2, gene_3=gene_3, gene_4=gene_4, gene_5=gene_5,
                gene_6=gene_6, gene_7=gene_7, gene_8=gene_8, gene_9=gene_9, gene_10=gene_10,
                snp_1=snp_1, snp_2=snp_2, snp_3=snp_3, snp_4=snp_4, snp_5=snp_5,
                age=age, bmi=bmi, smoking_status=smoking_status,
                protein_similarity=protein_similarity,
                predicted_result=ensemble_pred,
                prediction_confidence=ensemble_confidence,
                model_used='Ensemble'
            )
            
            log_activity(request.user, 'PREDICTION', f'Made genetic disorder prediction', request)
            
            context = {
                'predictions': predictions,
                'ensemble_prediction': ensemble_pred,
                'ensemble_confidence': ensemble_confidence,
                'input_data': {
                    'gene_1': gene_1, 'gene_2': gene_2, 'gene_3': gene_3, 'gene_4': gene_4,
                    'gene_5': gene_5, 'gene_6': gene_6, 'gene_7': gene_7, 'gene_8': gene_8,
                    'gene_9': gene_9, 'gene_10': gene_10, 'snp_1': snp_1, 'snp_2': snp_2,
                    'snp_3': snp_3, 'snp_4': snp_4, 'snp_5': snp_5, 'age': age,
                    'bmi': bmi, 'smoking_status': smoking_status,
                    'protein_similarity': protein_similarity
                }
            }
            
            return render(request, 'prediction_result.html', context)
            
        except Exception as e:
            messages.error(request, f'Prediction error: {str(e)}')
            return redirect('predict')
    
    return render(request, 'predict.html')

@login_required
def model_comparison_view(request):
    """Model comparison and visualization"""
    model_performances = ModelPerformance.objects.all().order_by('-accuracy')
    
    # Prepare data for charts
    models_data = []
    for mp in model_performances:
        models_data.append({
            'model': mp.model_name,
            'accuracy': mp.accuracy,
            'precision': mp.precision,
            'recall': mp.recall,
            'f1_score': mp.f1_score,
            'training_time': mp.training_time
        })
    
    context = {
        'model_performances': model_performances,
        'models_data_json': json.dumps(models_data)
    }
    
    return render(request, 'model_comparison.html', context)

@login_required
def data_preprocessing_view(request):
    """Data preprocessing information"""
    context = {
        'features': [
            {'name': 'gene_1 to gene_10', 'type': 'Continuous', 'description': 'Gene expression values'},
            {'name': 'snp_1 to snp_5', 'type': 'Binary', 'description': 'Single Nucleotide Polymorphisms'},
            {'name': 'age', 'type': 'Integer', 'description': 'Patient age in years'},
            {'name': 'bmi', 'type': 'Continuous', 'description': 'Body Mass Index'},
            {'name': 'smoking_status', 'type': 'Binary', 'description': '0: Non-smoker, 1: Smoker'},
            {'name': 'protein_similarity', 'type': 'Continuous', 'description': 'Protein sequence similarity index'},
        ],
        'preprocessing_steps': [
            'Data loading from CSV file',
            'Missing value handling',
            'Feature scaling using StandardScaler',
            'Train-test split (80-20)',
            'Stratified sampling to maintain class balance',
        ]
    }
    
    return render(request, 'data_preprocessing.html', context)

@login_required
def profile_view(request):
    """User profile view"""
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        # Update profile
        request.user.first_name = request.POST.get('first_name', '')
        request.user.last_name = request.POST.get('last_name', '')
        request.user.email = request.POST.get('email', '')
        request.user.save()
        
        profile.phone = request.POST.get('phone', '')
        profile.address = request.POST.get('address', '')
        
        if request.POST.get('date_of_birth'):
            profile.date_of_birth = request.POST.get('date_of_birth')
        
        if request.FILES.get('profile_image'):
            profile.profile_image = request.FILES['profile_image']
        
        profile.save()
        
        log_activity(request.user, 'PROFILE_UPDATE', 'Profile updated', request)
        messages.success(request, 'Profile updated successfully!')
        return redirect('profile')
    
    context = {
        'profile': profile,
    }
    
    return render(request, 'profile.html', context)

@login_required
def prediction_history_view(request):
    """View prediction history"""
    predictions = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')
    
    context = {
        'predictions': predictions
    }
    
    return render(request, 'prediction_history.html', context)

@login_required
def generate_report_view(request, prediction_id):
    """Generate detailed prediction report"""
    prediction = get_object_or_404(PredictionHistory, id=prediction_id, user=request.user)
    
    log_activity(request.user, 'REPORT_GENERATION', f'Generated report for prediction #{prediction_id}', request)
    
    context = {
        'prediction': prediction
    }
    
    return render(request, 'report.html', context)

# Admin views
@login_required
def admin_dashboard_view(request):
    """Admin dashboard"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('dashboard')
    
    # Statistics
    total_users = User.objects.count()
    total_predictions = PredictionHistory.objects.count()
    recent_activities = UserActivity.objects.all()[:20]
    
    # User registration stats
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    new_users_week = User.objects.filter(date_joined__gte=week_ago).count()
    
    context = {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'new_users_week': new_users_week,
        'recent_activities': recent_activities,
    }
    
    return render(request, 'admin/dashboard.html', context)

@login_required
def user_management_view(request):
    """User management for admin"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    
    users = User.objects.all().order_by('-date_joined')
    
    context = {
        'users': users
    }
    
    return render(request, 'admin/user_management.html', context)

@login_required
def add_user_view(request):
    """Add new user (admin)"""
    if not request.user.is_staff:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        is_admin = request.POST.get('is_admin') == 'on'
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
        else:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                is_staff=is_admin
            )
            UserProfile.objects.create(user=user)
            log_activity(request.user, 'USER_CREATION', f'Created user: {username}', request)
            messages.success(request, f'User {username} created successfully!')
            return redirect('user_management')
    
    return render(request, 'admin/add_user.html')

@login_required
def edit_user_view(request, user_id):
    """Edit user (admin)"""
    if not request.user.is_staff:
        return redirect('dashboard')
    
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        user.username = request.POST.get('username')
        user.email = request.POST.get('email')
        user.is_staff = request.POST.get('is_admin') == 'on'
        
        new_password = request.POST.get('new_password')
        if new_password:
            user.set_password(new_password)
        
        user.save()
        log_activity(request.user, 'USER_UPDATE', f'Updated user: {user.username}', request)
        messages.success(request, f'User {user.username} updated successfully!')
        return redirect('user_management')
    
    context = {
        'edit_user': user
    }
    
    return render(request, 'admin/edit_user.html', context)

@login_required
def delete_user_view(request, user_id):
    """Delete user (admin)"""
    if not request.user.is_staff:
        return redirect('dashboard')
    
    user = get_object_or_404(User, id=user_id)
    
    if user.id == request.user.id:
        messages.error(request, 'You cannot delete your own account!')
    else:
        username = user.username
        user.delete()
        log_activity(request.user, 'USER_DELETION', f'Deleted user: {username}', request)
        messages.success(request, f'User {username} deleted successfully!')
    
    return redirect('user_management')

@login_required
def user_activity_view(request):
    """View user activity logs (admin)"""
    if not request.user.is_staff:
        return redirect('dashboard')
    
    activities = UserActivity.objects.all().order_by('-timestamp')[:100]
    
    context = {
        'activities': activities
    }
    
    return render(request, 'admin/user_activity.html', context)
