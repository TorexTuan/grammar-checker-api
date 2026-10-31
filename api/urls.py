from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .auth_views import csrf_token, login_view, logout_view, me, register
from .document_views import DocumentViewSet
from .views import health_check, check_grammar

router = DefaultRouter()
router.register('documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('health/', health_check, name='health-check'),
    path('check/', check_grammar, name='check-grammar'),
    path('auth/csrf/', csrf_token, name='auth-csrf'),
    path('auth/register/', register, name='auth-register'),
    path('auth/login/', login_view, name='auth-login'),
    path('auth/logout/', logout_view, name='auth-logout'),
    path('auth/me/', me, name='auth-me'),
    path('', include(router.urls)),
]
