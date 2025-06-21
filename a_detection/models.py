from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class NewsArticle(models.Model):
    title = models.CharField(max_length=400, blank = True)
    content = models.TextField()
    date_uploaded = models.DateTimeField(auto_now_add=True)
    
    Choices = [("REAL","REAL"),
               ("FAKE","FAKE")]
    result = models.CharField(max_length=20,choices=Choices,default="REAL")
    confidence_score = models.DecimalField(decimal_places=2,max_digits=4)
    
    
    def __str__(self):
        return "Fake" if self.result == "FAKE" else "REAL"
    
    class Meta:
        ordering = ['-date_uploaded']
        verbose_name = "News Article"
        verbose_name_plural = "News Articles"
        
        
        
class UserFeedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    article = models.ForeignKey(NewsArticle, on_delete=models.CASCADE, related_name='feedbacks')
    message = models.TextField()
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

    class Meta:
        ordering = ['-submitted_at']
        verbose_name = "User Feedback"
        verbose_name_plural = "User Feedbacks"