from django.urls import path
from .views import landing_view, category_recipes_view

urlpatterns = [
    path('', landing_view, name='landing_page'),
    path('category/<str:category>/', category_recipes_view, name='category_recipes'),
]