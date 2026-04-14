#!/bin/bash

echo "================================================"
echo "Genetic Disorder Prediction System - Setup"
echo "================================================"
echo ""

# Install dependencies
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt --break-system-packages
echo "✓ Dependencies installed"
echo ""

# Database setup - CRITICAL STEP
echo "Step 2: Setting up database..."
echo "Creating migrations..."
python manage.py makemigrations app
echo ""
echo "Applying migrations..."
python manage.py migrate
echo "✓ Database created and migrations applied"
echo ""

# Create superuser
echo "Step 3: Creating admin user..."
echo "Please enter admin credentials:"
python manage.py createsuperuser
echo "✓ Admin user created"
echo ""

# Train models
echo "Step 4: Training ML/DL models (this may take 5-10 minutes)..."
python manage.py train_models
echo "✓ Models trained"
echo ""

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To start the server, run:"
echo "  python manage.py runserver"
echo ""
echo "Then open your browser to:"
echo "  http://127.0.0.1:8000"
echo ""
echo "IMPORTANT: Login with the admin credentials you just created!"
echo ""
echo "================================================"
