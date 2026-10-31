from django.contrib import admin

from .models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'owner', 'version', 'updated_at')
    search_fields = ('title', 'owner__email')
    list_filter = ('created_at', 'updated_at')

# Register your models here.
