from django.contrib import admin
from .models import NewsArticle,UserFeedback
# Register your models here.

class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ['result','confidence_score','date_uploaded']
admin.site.register(NewsArticle,NewsArticleAdmin)


class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ["user","message","submitted_at"]
admin.site.register(UserFeedback,UserFeedbackAdmin)