from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # User views
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('profile/', views.profile_view, name='profile'),
    path('prediction-history/', views.prediction_history_view, name='prediction_history'),
    path('report/<int:prediction_id>/', views.generate_report_view, name='generate_report'),
    
    # Model information
    path('model-comparison/', views.model_comparison_view, name='model_comparison'),
    path('data-preprocessing/', views.data_preprocessing_view, name='data_preprocessing'),
    
    # Admin views
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('user-management/', views.user_management_view, name='user_management'),
    path('add-user/', views.add_user_view, name='add_user'),
    path('edit-user/<int:user_id>/', views.edit_user_view, name='edit_user'),
    path('delete-user/<int:user_id>/', views.delete_user_view, name='delete_user'),
    path('user-activity/', views.user_activity_view, name='user_activity'),
]
