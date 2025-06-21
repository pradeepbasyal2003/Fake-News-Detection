from django import forms
from .models import UserFeedback

class UserFeedbackForm(forms.ModelForm):
    class Meta:
        model = UserFeedback
        fields = ['message']