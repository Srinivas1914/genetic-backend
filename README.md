# Genetic Disorder Prediction System
## Advanced ML/DL-based Predictive Analytics Platform

### 🧬 Overview
A comprehensive web-based system for predicting genetic disorders using multiple Machine Learning and Deep Learning models. Features user authentication, admin panel, model comparison, and detailed predictions.

### ✨ Features

#### User Features:
- User Registration & Login (username or email)
- Profile Management with image upload
- Genetic Disorder Prediction using 14 AI models
- Prediction History tracking
- Detailed Report Generation
- Model Performance Comparison
- Data Preprocessing Information

#### Admin Features:
- Admin Dashboard with statistics
- User Management (Add, Edit, Delete users)
- Role Management (User to Admin conversion)
- User Activity Logging
- Complete access to prediction features

#### ML/DL Models (14 models):
**Traditional ML:** KNN, Random Forest, SVM, Decision Tree, Logistic Regression, Gradient Boosting, XGBoost, LightGBM, Naive Bayes, AdaBoost, Extra Trees
**Deep Learning:** Neural Network, CNN, RNN-LSTM

### 📊 Input Features (19 features):
- Gene Expression: gene_1 to gene_10
- SNPs: snp_1 to snp_5
- Clinical: age, bmi, smoking_status, protein_similarity

### 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup database
python manage.py makemigrations
python manage.py migrate

# 3. Create admin user
python manage.py createsuperuser

# 4. Train models (5-10 minutes)
python manage.py train_models

# 5. Run server
python manage.py runserver

# 6. Open browser: http://127.0.0.1:8000
```

### 🎯 Model Performance
- KNN: 95%+ accuracy target
- Random Forest: 90%+ accuracy
- Other models: 85%+ accuracy

### 📁 Project Structure
```
genetic_disorder_prediction/
├── app/                    # Main application
│   ├── ml_models/         # Trained models
│   ├── templates/         # HTML templates
│   ├── ml_trainer.py      # Training module
│   ├── models.py          # Database models
│   └── views.py           # Views
├── genetic_data.csv       # Dataset
├── manage.py
├── requirements.txt
└── README.md
```

### 🔒 Security
- Password hashing
- CSRF protection
- Session management
- Activity logging with IP tracking

### 📧 Default Login
After creating superuser, login with your credentials.

---
Developed with Django, TensorFlow, and scikit-learn
