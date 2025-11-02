from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib.auth.forms import AuthenticationForm
from allauth.account.forms import LoginForm, SignupForm
from allauth.account.views import LoginView as AllauthLoginView, SignupView as AllauthSignupView
from allauth.account.utils import perform_login
from allauth.account import app_settings
from allauth.utils import get_form_class
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect


class CustomLoginView(AllauthLoginView):
    """Custom login view with enhanced error handling and success messages"""
    
    def form_valid(self, form):
        """Override form_valid to add success message"""
        # Call parent's form_valid method
        response = super().form_valid(form)
        
        # Add success message
        messages.success(
            self.request, 
            f"Welcome back, {self.request.user.username}! You have successfully logged in.",
            extra_tags='success'
        )
        
        return response
    
    def form_invalid(self, form):
        """Override form_invalid to add error messages"""
        # Add specific error messages based on form errors
        if 'login' in form.errors:
            messages.error(
                self.request,
                "Invalid email or password. Please check your credentials and try again.",
                extra_tags='error'
            )
        elif 'password' in form.errors:
            messages.error(
                self.request,
                "Invalid password. Please check your password and try again.",
                extra_tags='error'
            )
        else:
            messages.error(
                self.request,
                "Login failed. Please check your information and try again.",
                extra_tags='error'
            )
        
        return super().form_invalid(form)


class CustomSignupView(AllauthSignupView):
    """Custom signup view with enhanced error handling and success messages"""
    
    def form_valid(self, form):
        """Override form_valid to add success message"""
        # Call parent's form_valid method
        response = super().form_valid(form)
        
        # Add success message
        messages.success(
            self.request,
            f"Welcome to Fake News Detection! Your account has been created successfully. You are now logged in.",
            extra_tags='success'
        )
        
        return response
    
    def form_invalid(self, form):
        """Override form_invalid to add error messages"""
        # Add specific error messages based on form errors
        if 'email' in form.errors:
            if 'unique' in str(form.errors['email']):
                messages.error(
                    self.request,
                    "An account with this email already exists. Please use a different email or try logging in.",
                    extra_tags='error'
                )
            else:
                messages.error(
                    self.request,
                    "Please enter a valid email address.",
                    extra_tags='error'
                )
        elif 'password1' in form.errors:
            messages.error(
                self.request,
                "Password requirements not met. Please ensure your password is strong and meets all requirements.",
                extra_tags='error'
            )
        elif 'password2' in form.errors:
            messages.error(
                self.request,
                "Passwords do not match. Please ensure both password fields are identical.",
                extra_tags='error'
            )
        elif 'username' in form.errors:
            if 'unique' in str(form.errors['username']):
                messages.error(
                    self.request,
                    "This username is already taken. Please choose a different username.",
                    extra_tags='error'
                )
            else:
                messages.error(
                    self.request,
                    "Please enter a valid username.",
                    extra_tags='error'
                )
        else:
            messages.error(
                self.request,
                "Registration failed. Please check your information and try again.",
                extra_tags='error'
            )
        
        return super().form_invalid(form)


@require_http_methods(["GET", "POST"])
def custom_logout_view(request):
    """Custom logout view with success message"""
    if request.method == "POST":
        messages.success(
            request,
            "You have been successfully logged out. Thank you for using Fake News Detection!",
            extra_tags='success'
        )
        return redirect('account_logout')
    return redirect('home')


def handle_authentication_errors(request):
    """Handle common authentication errors and display appropriate messages"""
    if not request.user.is_authenticated:
        messages.warning(
            request,
            "You need to be logged in to access this feature. Please sign in or create an account.",
            extra_tags='warning'
        )
        return redirect('account_login')
    return None
