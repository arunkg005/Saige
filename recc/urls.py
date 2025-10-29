from django.urls import path
from .views import recc_page_view

urlpatterns = [
    path('', recc_page_view, name='recc_page'),
]
