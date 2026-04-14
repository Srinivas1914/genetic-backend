# Quick Start Guide - Genetic Disorder Prediction System

## 🚀 Get Started in 3 Minutes!

### Step 1: Extract Files
Extract the ZIP file to your desired location.

### Step 2: Install & Setup
Open terminal/command prompt in the project folder:

**Option A - Automated Setup (Recommended):**
```bash
python setup.py
```

**Option B - Manual Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database
python manage.py makemigrations
python manage.py migrate

# Create admin account
python manage.py createsuperuser
```

### Step 3: Run Server
```bash
python manage.py runserver
```

### Step 4: Access Application
Open browser: http://127.0.0.1:8000/

## 📝 First Time Usage

### For Users:
1. Click "Register Now" on login page
2. Fill in: username, email, password
3. Login with your credentials
4. Make predictions with 19 input features
5. View results from 12+ ML/DL models
6. Download PDF reports

### For Admins:
1. Login with superuser credentials
2. Access Admin menu → Admin Dashboard
3. Manage users (add/edit/delete)
4. View activity logs
5. Also make predictions like regular users

## 🎯 Quick Test

**Auto-fill Test Data:**
- On prediction form, press `Ctrl+R` to auto-fill with random test values
- Click "Predict with All Models"
- View instant results from all algorithms!

## 📊 What You Get

### Input Features (19):
- **Genes**: gene_1 to gene_10 (10 features)
- **SNPs**: snp_1 to snp_5 (5 features)
- **Clinical**: age, bmi, smoking_status, protein_similarity (4 features)

### ML/DL Models (12+):
1. ✅ K-Nearest Neighbors (KNN) - **95%+ accuracy**
2. Random Forest
3. Support Vector Machine (SVM)
4. Decision Tree
5. Gradient Boosting
6. XGBoost
7. LightGBM
8. CatBoost
9. Logistic Regression
10. Multi-Layer Perceptron (MLP)
11. CNN (Deep Learning)
12. LSTM (Deep Learning)

### Features:
- ✨ Real-time predictions with all models
- 📊 Visual comparisons & graphs
- 📄 Comprehensive PDF reports
- 👥 User management (Admin)
- 📈 Activity tracking
- 🔐 Secure authentication

## 🎨 User Interface Highlights

- **Beautiful Login Page**: Modern, responsive design
- **Dynamic Dashboard**: Stats, prediction form, recent history
- **Model Comparisons**: Interactive charts showing all model performances
- **Data Info**: Complete preprocessing pipeline details
- **Admin Panel**: Powerful user management tools

## ⚡ Performance

- **Best Model**: KNN with 95%+ accuracy
- **All Models**: Above 85% accuracy
- **Dataset**: 48,119 samples
- **Training Time**: 5-15 minutes (first run only)
- **Prediction Time**: < 2 seconds for all models

## 🔧 Troubleshooting

**Issue: Import errors**
```bash
pip install -r requirements.txt --upgrade
```

**Issue: Database errors**
```bash
rm db.sqlite3
python manage.py migrate
```

**Issue: Port already in use**
```bash
python manage.py runserver 8080
```
Then access: http://127.0.0.1:8080/

## 📚 Navigation Guide

### Main Pages:
- **Dashboard**: Make predictions, view stats
- **Comparisons**: See all model performance metrics
- **Data Info**: Dataset and preprocessing details
- **Profile**: Edit your account details

### Admin Pages:
- **Admin Dashboard**: System overview
- **User Management**: Add/edit/delete users
- **Activity Logs**: View all user actions

## 💡 Pro Tips

1. **Fast Testing**: Use Ctrl+R for random test data
2. **Best Results**: KNN model has highest accuracy
3. **Reports**: Download PDF reports for record keeping
4. **Comparisons**: Check model comparisons for insights
5. **Role Management**: Admins can change user roles anytime

## 🎓 Understanding Results

### Prediction Output:
- **Disorder Detected** (Red): Positive prediction
- **No Disorder** (Green): Negative prediction
- **Ensemble Prediction**: Final verdict from all models
- **Confidence**: Percentage of models agreeing

### Metrics:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: Finding all positive cases
- **F1-Score**: Balance of precision and recall

## 🔐 Default Admin Account

Create during setup with:
```bash
python manage.py createsuperuser
```

Remember your credentials - you'll need them to access admin features!

## 📞 Support

For issues:
1. Check README.md for detailed documentation
2. Verify all dependencies are installed
3. Ensure Python 3.8+ is being used
4. Check database migrations are complete

## 🎉 You're Ready!

The system is now ready to use. Start making predictions and exploring the powerful ML/DL models!

**Happy Predicting! 🧬🔬**

---

*For detailed documentation, see README.md*
