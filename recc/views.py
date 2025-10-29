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
BASE_DIR = Path(__file__).resolve().parent.parent.parent # Adjust if views.py location changes
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
        prompt = f"""
        Convert the following user input into a JSON dictionary with these keys:
        'Diet_Type' (choose from ['Vegan', 'Veg', 'Keto', 'Non-Veg', 'any']),
        'Essential_Ingredients' (list of lowercase strings),
        'Other_Ingredients' (list of lowercase strings),
        'Taste_Profile' (choose from ['Spicy', 'Savory', 'Sour', 'Mild', 'Sweet', 'Neutral', 'Bitter', 'any'])

        Rules:
        - Return ONLY a valid JSON object. No extra text before or after.
        - If a signal is not mentioned, use "any" for Diet_Type and Taste_Profile, and an empty list [] for ingredients.
        - "Essential_Ingredients" are main foods (e.g., chicken, potato, rice). Convert to lowercase.
        - "Other_Ingredients" are spices, herbs, secondary items (e.g., curry leaves, garlic). Convert to lowercase.
        - Be concise with ingredient names (e.g., "onion" not "one medium onion").

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

            return render(request, "recc_page.html", context)
        else:
             # Should not happen if error handling above is correct, but just in case
             context["fallback_msg"] = "Failed to get structured query from AI model."
             return render(request, "recc_page.html", context)

    # Handle GET request (show the empty page)
    return render(request, "recc_page.html", context)