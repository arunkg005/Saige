from django.shortcuts import render
import os
import json
from django.conf import settings # If you decide to use Django settings later
from django.http import JsonResponse, HttpRequest # Added HttpRequest for type hinting
from pathlib import Path
# Assuming your pipeline module is correctly placed
# Adjust the import path if needed (e.g., from ..pipeline.recipe_pipeline import ...)
from .saige_model.saige_m1 import get_recommendations
import google.generativeai as genai

# --- Load API key from .env manually ---
# (This loading logic seems correct based on your previous code)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'saige'  # Explicitly set BASE_DIR to the 'saige' folder
ENV_PATH = BASE_DIR / 'saige' / '.env'
if not ENV_PATH.exists():
    ENV_PATH = BASE_DIR / '.env' # Check root

GEMINI_API_KEY = None
try:
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Ensure this key name matches EXACTLY what's in your .env file
                if key == 'saige_key':
                    GEMINI_API_KEY = value.strip().replace('"', '').replace("'", "")
                    break
except FileNotFoundError:
    print(f"--- [views.py ERROR]: .env file not found at {ENV_PATH} or {BASE_DIR / '.env'} ---")
    # You might want to raise an ImproperlyConfigured exception here in production
except Exception as e:
    print(f"--- [views.py ERROR] reading .env file: {e} ---")

if not GEMINI_API_KEY:
    print("--- [views.py WARNING]: Could not find 'saige_key' in .env file. API calls might fail. ---")
    # Decide how to handle this - maybe raise an error or use a default?

# --- Configure the API ---
# This only runs once when Django starts
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("--- [views.py INFO]: Gemini API configured successfully. ---")
    else:
        # Don't try to configure if key wasn't found
        print("--- [views.py ERROR]: Cannot configure Gemini API - Key not loaded. ---")

except Exception as e:
    print(f"--- [views.py ERROR]: Error configuring Gemini API: {e} ---")
    # Handle error appropriately (e.g., log it)

# --- Define Permissive Safety Settings ---
# This prevents blocking on common recipe terms
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# === Your Main Django View ===
def recc_page_view(request: HttpRequest):
    context = {
        "recommendations": [],
        "fallback_msg": "",
        "search_text": ""
    }

    if request.method == "POST":
        user_text = request.POST.get('search_text', '').strip()
        context["search_text"] = user_text # Keep search text in context

        if not user_text:
            context["fallback_msg"] = "Please enter some ingredients or preferences."
            return render(request, "recc_page.html", context)

        if not GEMINI_API_KEY:
            context["fallback_msg"] = "API Key not configured. Cannot process request."
            return render(request, "recc_page.html", context)

        # --- Call Gemini API to get signals ---
        # --- Updated Prompt to include Dish Name logic ---
        prompt = f"""
            Convert the following user input into a JSON dictionary with these keys:
            'Diet_Type' (choose from ['Vegan', 'Veg', 'Keto', 'Non-Veg', 'any']),
            'Essential_Ingredients' (list of lowercase strings ),
            'Other_Ingredients' (list of lowercase strings),
            'Taste_Profile' (choose from ['Spicy', 'Savory', 'Sour', 'Mild', 'Sweet', 'Neutral', 'Bitter', 'any']),
            'Dish_Name' (return the correct main dish name if the user input clearly specifies a dish they want to eat, with proper spelling. Only return if the dish name is explicitly mentioned or clearly implied. Do not return a dish name if it is used as a reference for taste or merged with ingredient names.)

            VERY IMPORTANT RULES:

            1. **Agar user sirf dish ka naam likhe** (jaise "biryani", "pav bhaji", "rajma chawal"):
            - Tab bhi us dish ke common / typical ingredients ka best guess do.
            - "Essential_Ingredients" me main cheezein (jaise "rice", "chicken", "potato").
            - "Other_Ingredients" me spices, herbs, aur helpers (jaise "garam masala", "ginger", "garlic").

            2. **Kabhi bhi sirf dish name ke saath empty ingredient lists mat bhejo.**
            - Agar dish name mila hai, to "Essential_Ingredients" aur "Other_Ingredients" dono me kuch na kuch items zaroor hone chahiye.
            - Sirf us case me [] use karo jab na dish ka naam ho, na ingredients ka koi hint ho.

            3. "Essential_Ingredients":
            - Sirf main food items jo dish banane ke liye zaroori hote hain.
            - Example: "rice", "chicken", "potato", "paneer".
            - Sab lowercase me.

            4. "Other_Ingredients":
            - Spices, herbs, aur optional / secondary items.
            - Example: "cumin seeds", "garam masala", "coriander leaves", "ginger", "garlic".
            - Sab lowercase me.

            5. "Taste_Profile":
            - Agar user ne spicy / sweet / tangy waghara mention kiya ho to uska best match select karo.
            - Agar kuch clear nahi hai, to "any" use karo.

            6. "Dish_Name":
            - Agar user clearly kisi dish ko eat / cook karna chah raha hai (e.g., "I want to eat biryani today" ya "bana pav bhaji"), to proper spelled dish name string me do.
            - Agar dish sirf example ke liye mention hai (jaise "I don't want biryani"), to tab "Dish_Name" ko null (ya empty string) rakho.

            7. Output format:
            - Sirf ek valid JSON object return karo.
            - JSON ke bahar koi extra text, explanation, comments, ya markdown mat likho.

            Example 1:
            User input: "biryani"
            Possible JSON (approx):
            {{
            "Diet_Type": "any",
            "Essential_Ingredients": ["rice", "onion", "tomato"],
            "Other_Ingredients": ["garam masala", "ginger", "garlic"],
            "Taste_Profile": "Spicy",
            "Dish_Name": "Biryani"
            }}

            Example 2:
            User input: "make veg biryani with less oil"
            Possible JSON (approx):
            {{
            "Diet_Type": "Veg",
            "Essential_Ingredients": ["rice", "mixed vegetables"],
            "Other_Ingredients": ["garam masala", "ginger", "garlic"],
            "Taste_Profile": "Spicy",
            "Dish_Name": "Veg Biryani"
            }}
            

            Now process this input:

            User input: "{user_text}"
            JSON Output:
            """

        model_output = None # Initialize variable
        user_query = None

        try:
            # 1. Initialize the model
            # !!! PASTE THE CORRECT MODEL NAME FROM check_models.py HERE !!!
            model_name_from_listmodels = 'models/gemini-2.0-flash' # <-- REPLACE THIS PLACEHOLDER
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            model = genai.GenerativeModel(
                model_name_from_listmodels,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )

            # 2. Call the API with safety settings
            print(f"--- [views.py INFO]: Calling Gemini with prompt for: '{user_text}' ---")
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings
            )

            # 3. Improved Error Checking for Blocks/Empty Response
            if not response.parts:
                block_reason_msg = ""
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason_msg = f" Reason: {response.prompt_feedback.block_reason}"
                error_msg = f"Gemini request failed (empty response).{block_reason_msg}"
                print(f"--- [views.py ERROR]: {error_msg} ---")
                context["fallback_msg"] = error_msg
                return render(request, "recc_page.html", context)

            # 4. Get text and parse JSON
            model_output = response.text # Get the raw text first for debugging
            print(f"--- [views.py DEBUG]: Raw Gemini Output: {model_output} ---")
            user_query = json.loads(model_output) # Now parse

            # --- Basic Validation of the received JSON ---
            required_keys = ['Diet_Type', 'Essential_Ingredients', 'Other_Ingredients', 'Taste_Profile']
            if not all(key in user_query for key in required_keys):
                raise ValueError(f"Model output missing required keys. Got: {user_query}")
            if not isinstance(user_query['Essential_Ingredients'], list) or not isinstance(user_query['Other_Ingredients'], list):
                 raise ValueError(f"Ingredients are not lists. Got: {user_query}")


        except json.JSONDecodeError as e:
            error_msg = f"Error parsing model output (invalid JSON): {e}. Raw output: '{model_output}'"
            print(f"--- [views.py ERROR]: {error_msg} ---")
            context["fallback_msg"] = error_msg
            return render(request, "recc_page.html", context)

        except Exception as e:
            # Catch other errors (e.g., 404 model not found, API key invalid, network issues)
            error_msg = f"An error occurred calling the AI model: {e}"
            print(f"--- [views.py ERROR]: {error_msg} ---")
            context["fallback_msg"] = error_msg
            return render(request, "recc_page.html", context)

        # --- Call your recipe pipeline ---
        if user_query:
            try:
                print(f"--- [views.py INFO]: Calling recipe pipeline with signals: {user_query} ---")
                recommendations, fallback_msg = get_recommendations(user_query, top_n=5)
                context["recommendations"] = recommendations
                context["fallback_msg"] = fallback_msg
                print(f"--- [views.py INFO]: Pipeline returned {len(recommendations)} recommendations. ---")
            except Exception as e:
                # Catch errors from your recommendation pipeline itself
                error_msg = f"Error running recommendation pipeline: {e}"
                print(f"--- [views.py ERROR]: {error_msg} ---")
                context["fallback_msg"] = error_msg

            # --- Search Mechanism for Dish Name in CSV ---
            import pandas as pd  # Ensure pandas is imported

            # Load the CSV file once (can be optimized further if needed)
            csv_path = BASE_DIR / 'recc' / 'saige_model' / 'clustered_recipes.csv'  # Update the CSV path to use the corrected BASE_DIR
            recipes_df = pd.read_csv(csv_path)

            # Initialize search_results to avoid UnboundLocalError
            search_results = []

            dish_name = user_query.get('Dish_Name')
            if dish_name:
                dish_name = dish_name.strip().lower()
                # Perform partial match instead of exact match
                search_results = recipes_df[recipes_df['TranslatedRecipeName'].str.contains(dish_name, case=False, na=False)].to_dict('records')

            # Add search results to context if any matches are found
            if search_results:
                context['search_results'] = search_results

            # --- Prioritize Search Results in Recommendations ---
            if search_results:
                # If search results are found, show them first
                context['recommendations'] = search_results + context['recommendations']

            return render(request, "recc_page.html", context)
        else:
             # Should not happen if error handling above is correct, but just in case
             context["fallback_msg"] = "Failed to get structured query from AI model."
             return render(request, "recc_page.html", context)

    # Handle GET request (show the empty page)
    return render(request, "recc_page.html", context)

# Debugging the .env file path and API key loading
print(f"--- [DEBUG] ENV_PATH: {ENV_PATH} ---")
