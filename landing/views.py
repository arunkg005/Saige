from django.shortcuts import render
from django.core.paginator import Paginator

def landing_view(request):
    return render(request, "landing_page.html")

def category_recipes_view(request, category):
    # Dummy data for demonstration; replace with real data source
    recipes = [
        {
            "name": f"{category.title()} Recipe {i+1}",
            "image_url": f"/static/images/landing_img/{category}.jpg",
            "recipe_url": f"https://www.example.com/{category}-recipe-{i+1}"
        }
        for i in range(100)
    ]
    paginator = Paginator(recipes, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    context = {
        "category_name": category.title(),
        "recipes": page_obj.object_list,
        "page_obj": page_obj
    }
    return render(request, "category_recipes.html", context)