# saige_m1.py

import pandas as pd
import pickle
import numpy as np
import ast
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- GLOBAL: Determine Script Directory ---
# Find the absolute path to the directory where this script lives.
# This is crucial for reliably finding the asset files (CSV, PKL).
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"--- [DEBUG] SCRIPT_DIR resolved to: {SCRIPT_DIR} ---") # Debug print

# --- [HELPER FUNCTIONS] ---

# --- Step 2: Diet Filter ---
def _diet_tree_filter(dataframe, user_diet):
    """Filters the recipe DataFrame based on the user's diet preference (e.g., 'Vegan', 'Veg')."""
    if not user_diet or user_diet.lower() == 'any':
        # If no diet specified or 'any', return all recipes.
        return dataframe.copy()
    user_diet_lower = user_diet.lower()
    # Perform case-insensitive matching on the 'Diet_Type' column.
    return dataframe[dataframe['Diet_Type'].str.lower() == user_diet_lower].copy()

# --- Step 4.1: Cluster Selection ---
def _get_relevant_clusters(dataframe, all_user_ingredients):
    """Identifies recipe clusters relevant to the user's ingredients.
    This creates the 'collective area' to speed up the KNN search.
    """
    if 'ingredients_list' not in dataframe.columns or dataframe['ingredients_list'].isnull().all():
        print("  [Step 4.1] Warning: 'ingredients_list' column missing or empty. Cannot select clusters.")
        return None # Fallback: search all recipes if ingredients can't be checked.

    try:
        # 'Explode' converts rows with lists of ingredients into multiple rows, one per ingredient.
        df_exploded = dataframe.explode('ingredients_list')
        df_exploded = df_exploded.dropna(subset=['ingredients_list']) # Clean up potential errors
        # Find all rows (recipes) that contain any of the user's ingredients.
        matching_recipes = df_exploded[df_exploded['ingredients_list'].isin(all_user_ingredients)]
        # Get the unique cluster IDs associated with these matching recipes (ignore noise cluster -1).
        relevant_clusters = set(matching_recipes[matching_recipes['Cluster'] != -1]['Cluster'])

        if not relevant_clusters:
            print("  [Step 4.1] No specific clusters found containing the ingredients. Will search all.")
            return None # Fallback: search all recipes if no relevant clusters found.
        print(f"  [Step 4.1] Found {len(relevant_clusters)} relevant clusters: {relevant_clusters}")
        return relevant_clusters
    except Exception as e:
        print(f"  [Step 4.1] Error during cluster selection: {e}. Proceeding without cluster filtering.")
        return None # Fallback on error.


# --- Step 4.2: Weighted Query Vector ---
def _create_weighted_query_vector(essential_ingredients, other_ingredients, vectorizer, essential_weight=3):
    """Creates a TF-IDF vector representing the user's query, giving higher weight to essential ingredients."""
    print("  [Step 4.2] Creating weighted query vector...")
    essential_ingredients = [str(ing) for ing in essential_ingredients] # Ensure strings
    other_ingredients = [str(ing) for ing in other_ingredients] # Ensure strings
    # Simple weighting: Repeat essential ingredients multiple times in the query text.
    weighted_query_list = (essential_ingredients * essential_weight) + other_ingredients
    query_string = ' '.join(weighted_query_list)
    print(f"    Weighted query string (first 100 chars): '{query_string[:100]}...'")
    # Use the pre-loaded TF-IDF vectorizer to convert the text query into a numerical vector.
    query_vector = vectorizer.transform([query_string])
    # Normalize the vector (important for cosine similarity).
    return normalize(query_vector)

# --- Step 4.3: KNN Similarity Search (with Threshold) ---
def _find_similar_recipes(dataframe, tfidf_matrix, query_vector, relevant_clusters, top_n=20, similarity_threshold=0.1):
    """Finds recipes similar to the query vector using cosine similarity (KNN).
    Only returns recipes with similarity above the specified threshold.
    Initially fetches up to 'top_n' candidates before taste filtering.
    """
    print(f"  [Step 4.3] Running KNN search (Threshold: {similarity_threshold}, Max Candidates: {top_n})...")

    # Determine the subset of recipes to search within.
    if relevant_clusters:
        if 'Cluster' not in dataframe.columns:
             print("  [Step 4.3] Warning: 'Cluster' column not found. Searching all recipes.")
             candidate_df = dataframe
        else:
             # Search only within the recipes belonging to the relevant clusters.
             candidate_df = dataframe[dataframe['Cluster'].isin(relevant_clusters)]
    else:
        # If no relevant clusters, search the entire (already diet-filtered) dataframe.
        candidate_df = dataframe

    if candidate_df.empty:
        print("  [Step 4.3] No candidate recipes found.")
        return pd.DataFrame()

    candidate_indices = candidate_df.index
    # Safety check: Ensure indices exist in the global TF-IDF matrix.
    valid_indices = [idx for idx in candidate_indices if idx < tfidf_matrix.shape[0]]
    if len(valid_indices) != len(candidate_indices):
        print(f"  [Step 4.3] Warning: Some candidate indices out of bounds. Using {len(valid_indices)} valid indices.")
        candidate_indices = valid_indices
        if not candidate_indices:
             print("  [Step 4.3] No valid indices remaining.")
             return pd.DataFrame()
        candidate_df = dataframe.loc[candidate_indices] # Update candidate_df if indices changed

    # Select the rows from the main TF-IDF matrix corresponding to our candidates.
    candidate_matrix = tfidf_matrix[candidate_indices]
    # Calculate cosine similarity between the user's query vector and all candidate recipe vectors.
    cosine_sim = (candidate_matrix * query_vector.T).toarray().flatten()

    # --- Apply Similarity Threshold ---
    # Find the indices (within the candidate subset) where similarity meets the threshold.
    valid_indices_in_subset = np.where(cosine_sim >= similarity_threshold)[0]

    if len(valid_indices_in_subset) == 0:
        print(f"  [Step 4.3] No recipes met the similarity threshold of {similarity_threshold}.")
        return pd.DataFrame() # Return empty if no matches are good enough

    # Get the scores and the *original* DataFrame indices only for these valid matches.
    valid_scores = cosine_sim[valid_indices_in_subset]
    valid_original_indices = np.array(candidate_indices)[valid_indices_in_subset]

    # Sort the valid matches by score (highest first) and take the top N.
    num_results_to_take = min(top_n, len(valid_scores))
    # argsort gives indices that *would* sort ascending; slice last N, reverse for descending.
    top_indices_within_valid = valid_scores.argsort()[-num_results_to_take:][::-1]

    # Get the final original indices and their corresponding scores.
    final_original_indices = valid_original_indices[top_indices_within_valid]
    final_scores = valid_scores[top_indices_within_valid]
    # --- End Threshold Logic ---

    # Retrieve the full recipe details from the original DataFrame using the final indices.
    results_df = dataframe.loc[final_original_indices].copy()
    # Add the similarity score as a column to the results.
    results_df['similarity_score'] = final_scores
    print(f"  [Step 4.3] Found {len(results_df)} recipes above threshold.")
    return results_df


# --- Step 5: Taste Profile Filter ---
def _taste_profile_filter(knn_results_df, user_taste, final_n=10):
    """Filters KNN results by taste, implementing a 'Filter, then Fill' strategy.
    Aims to return 'final_n' recipes, prioritizing exact taste matches.
    Ensures results remain sorted by similarity score.
    """
    print(f"\n[Step 5] Applying 'Filter, then Fill' for taste '{user_taste}' (Targeting {final_n} results)")

    # If no results from KNN, or no taste specified, just return the top N KNN results.
    if knn_results_df.empty or not user_taste or user_taste.lower() == 'any':
        print("  [Step 5] No taste filter applied. Returning top results by similarity.")
        return knn_results_df.head(final_n), ""

    user_taste_lower = user_taste.lower()

    # Ensure KNN results are sorted by similarity *before* filtering.
    knn_results_df = knn_results_df.sort_values(by='similarity_score', ascending=False)

    # Separate recipes into exact taste matches and others.
    taste_matches = knn_results_df[knn_results_df['Taste_Profile'].str.lower() == user_taste_lower]
    other_matches = knn_results_df[knn_results_df['Taste_Profile'].str.lower() != user_taste_lower]

    fallback_message = ""

    # Apply "Filter, then Fill" logic
    if len(taste_matches) >= final_n:
        # Case 1: More than enough exact matches found. Return the top N.
        print(f"  [Step 5] Found {len(taste_matches)} exact matches. Returning top {final_n}.")
        final_df = taste_matches.head(final_n)
    elif len(taste_matches) > 0:
        # Case 2: Some exact matches, but not enough. Fill remaining slots with other matches.
        n_to_fill = final_n - len(taste_matches)
        print(f"  [Step 5] Found {len(taste_matches)} exact matches. Filling with {n_to_fill} nearby options.")
        other_matches_to_fill = other_matches.head(n_to_fill)
        # Combine exact matches and fill-in recipes.
        final_df = pd.concat([taste_matches, other_matches_to_fill]).head(final_n)
    else:
        # Case 3: No exact matches found. Fallback to returning top N other matches.
        print(f"  [Step 5] No exact match for '{user_taste}'. Falling back to nearby options.")
        fallback_message = f"No exact match found for '{user_taste}'. Showing other taste recipes."
        final_df = other_matches.head(final_n)

    # Ensure the final list maintains the original similarity score order.
    final_df = final_df.sort_values(by='similarity_score', ascending=False)

    return final_df, fallback_message


# --- Step 6: Output Formatting ---
def _format_recommendations(results_df, user_query, fallback_msg):
    """Formats the final list of recipes into the desired output structure (list of dicts).
    Adds a 'reason_for_recommendation' string explaining the match quality.
    """
    recommendations = []
    # Safely get taste/diet preferences from the query, defaulting to 'any'.
    user_taste = user_query.get("Taste_Profile", "any") or "any"
    user_diet = user_query.get("Diet_Type", "any") or "any"
    user_taste_lower = user_taste.lower()

    # Iterate through the final DataFrame (already sorted and sized correctly).
    for _, row in results_df.iterrows():
        # Calculate individual components for the 'reason' string.
        overlap_pct = f"{row['similarity_score'] * 100:.1f}% ingredient overlap"

        if user_diet == 'any':
             diet_match = "Diet (Not specified)"
        else:
            diet_match = f"100% Diet match ({row['Diet_Type']})"

        # Determine the taste match description based on fallback status and actual taste.
        row_taste_lower = row['Taste_Profile'].lower()
        if fallback_msg: # Case 3 from taste filter
            taste_match = f"Taste fallback (User wanted '{user_taste}', got '{row['Taste_Profile']}')"
        elif user_taste_lower == row_taste_lower: # Case 1 or exact match part of Case 2
            taste_match = f"100% Taste match ({row['Taste_Profile']})"
        else: # Fill-in recipe part of Case 2
            taste_match = f"Nearby taste (User wanted '{user_taste}', got '{row['Taste_Profile']}')"

        # Build the dictionary for this recommendation with required fields.
        rec_dict = {
            "Name": row['TranslatedRecipeName'],
            "diet": row['Diet_Type'],
            "taste": row['Taste_Profile'],
            "essential_ingredients": row.get('essential_ingredients_list', []), # Use .get for robustness
            "image_url": row.get('image-url', ''), # Use .get for robustness
            "recipe_url": row.get('URL', ''), # Use .get for robustness
            "total_time_mins": row.get('TotalTimeInMins', 0), # Use .get for robustness
            "reason_for_recommendation": f"{overlap_pct} | {diet_match} | {taste_match}"
        }
        recommendations.append(rec_dict)

    return recommendations, fallback_msg

# --- Load Assets Function (Using Absolute Paths) ---
def _load_all_assets():
    """Loads all pipeline assets (CSV, PKL files) from disk using absolute paths
    relative to this script's location. This runs only ONCE when the module is imported.
    """
    print(f"--- [recipe_pipeline] Attempting to load assets from directory: {SCRIPT_DIR} ---")

    clustered_recipes_file = SCRIPT_DIR / 'clustered_recipes.csv'
    vectorizer_file = SCRIPT_DIR / 'tfidf_vectorizer.pkl'
    tfidf_matrix_file = SCRIPT_DIR / 'tfidf_matrix.pkl'

    # Debug prints to show exactly where it's looking.
    print(f"--- [DEBUG] Checking for CSV at: {clustered_recipes_file} ---")
    print(f"--- [DEBUG] Checking for Vectorizer PKL at: {vectorizer_file} ---")
    print(f"--- [DEBUG] Checking for Matrix PKL at: {tfidf_matrix_file} ---")

    files = {
        "CSV": clustered_recipes_file,
        "Vectorizer": vectorizer_file,
        "Matrix": tfidf_matrix_file
    }
    missing_files = []
    # Check if each required file exists at the calculated path.
    for name, f_path in files.items():
        if not f_path.exists():
            missing_files.append(f"{name} (Path checked: {f_path})")

    # If any files are missing, print an error and stop loading.
    if missing_files:
        print("--- [recipe_pipeline] FATAL ERROR: Missing required files: ---")
        for mf in missing_files:
            print(f"  - {mf}")
        print("-------------------------------------------------------------")
        print(f"Ensure these files exist at the paths shown above.")
        return None # Return None to indicate failure.

    # If all files exist, attempt to load them.
    try:
        # Load the main recipe data CSV.
        df = pd.read_csv(clustered_recipes_file)
        # Check if essential columns are present in the CSV.
        required_cols = ['ingredients_list', 'essential_ingredients_list', 'TranslatedRecipeName', 'Diet_Type', 'Taste_Profile', 'image-url', 'URL', 'TotalTimeInMins', 'Cluster']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             print(f"--- [recipe_pipeline] FATAL ERROR: CSV is missing required columns: {missing_cols} ---")
             return None

        # Convert string representations of lists back into actual Python lists.
        df['ingredients_list'] = df['ingredients_list'].apply(ast.literal_eval)
        df['essential_ingredients_list'] = df['essential_ingredients_list'].apply(ast.literal_eval)
        print(f"Loaded '{clustered_recipes_file.name}'")

        # Load the saved TF-IDF vectorizer model.
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"Loaded '{vectorizer_file.name}'")

        # Load the saved TF-IDF matrix (recipe vectors).
        with open(tfidf_matrix_file, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        print(f"Loaded '{tfidf_matrix_file.name}'")

        print("--- [recipe_pipeline] All assets loaded successfully. ---")

        # Return a dictionary containing the loaded assets.
        return {
            "dataframe": df,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix
        }
    except Exception as e:
        # Catch any errors during file loading (e.g., corrupted file).
        print(f"--- [recipe_pipeline] An error occurred during asset loading (after finding files): {e} ---")
        return None

# --- [GLOBAL VARIABLE Initialization] ---
# Execute the loading function immediately when this Python module is first imported.
# The result (the dictionary of assets or None) is stored globally.
ASSETS = _load_all_assets()

# --- [MASTER FUNCTION - Entry Point for Recommendations] ---
# Default number of results to return is now 10.
def get_recommendations(user_query, top_n=10):
    """
    Runs the full recommendation pipeline: Diet Filter -> Cluster Selection ->
    Weighted KNN (with threshold) -> Taste Filter (Filter/Fill) -> Formatting.
    Returns 'top_n' recommendations sorted by ingredient similarity.
    """
    # --- Input Validation ---
    if not isinstance(user_query, dict):
        return [], "Error: Invalid user query format (must be a dictionary)."
    required_keys = ['Diet_Type', 'Essential_Ingredients', 'Other_Ingredients', 'Taste_Profile']
    if not all(key in user_query for key in required_keys):
         return [], f"Error: User query missing required keys ({required_keys}). Found: {list(user_query.keys())}"

    print(f"\n--- [recipe_pipeline] Received new query for: {user_query.get('Essential_Ingredients', [])} ---")

    # --- Check if Assets Loaded ---
    # Critical check: If assets didn't load during import, cannot proceed.
    if ASSETS is None:
        print("--- [recipe_pipeline] Error inside get_recommendations: Pipeline assets are not loaded. ---")
        return [], "Error: Pipeline assets could not be loaded."

    # Retrieve loaded assets.
    df = ASSETS["dataframe"]
    vectorizer = ASSETS["vectorizer"]
    tfidf_matrix = ASSETS["tfidf_matrix"]

    # --- Execute Pipeline Steps ---

    # Step 2: Filter by Diet
    diet_filtered_df = _diet_tree_filter(df, user_query["Diet_Type"])
    if diet_filtered_df.empty:
        # If no recipes match the diet, stop here.
        return [], f"Sorry, no recipes found matching the diet preference: '{user_query['Diet_Type']}'."

    # Step 4: Ingredient Matching (Cluster Selection + Weighted KNN)
    all_ingredients = user_query.get("Essential_Ingredients", []) + user_query.get("Other_Ingredients", [])
    if not all_ingredients:
         # If user provided no ingredients, stop here.
         print("[Step 4] No ingredients provided in the query. Skipping ingredient-based search.")
         return [], "Please provide some ingredients for matching."

    # 4.1: Find relevant clusters (optional speedup).
    relevant_clusters = _get_relevant_clusters(diet_filtered_df, all_ingredients)
    # 4.2: Create the weighted query vector.
    query_vector = _create_weighted_query_vector(
        user_query["Essential_Ingredients"],
        user_query["Other_Ingredients"],
        vectorizer
    )
    # 4.3: Perform KNN search within candidates, applying the similarity threshold.
    # Fetch more candidates (20) than finally needed (10) to allow for taste filtering.
    knn_results_df = _find_similar_recipes(
        diet_filtered_df,
        tfidf_matrix,
        query_vector,
        relevant_clusters,
        top_n=20, # Fetch more candidates
        similarity_threshold=0.1 # Minimum ingredient overlap required
    )
    print(f"[Step 4] Found {len(knn_results_df)} initial KNN matches above threshold.")

    # If KNN found no recipes meeting the similarity threshold, stop here.
    if knn_results_df.empty:
        return [], "Sorry, no recipes found with a close enough ingredient match."

    # Step 5: Filter/Fill based on Taste preference.
    final_results_df, fallback_msg = _taste_profile_filter(
        knn_results_df,
        user_query["Taste_Profile"],
        final_n=top_n # Aim for the final requested number (default 10)
    )
    print(f"[Step 5] Filtered/Filled list. {len(final_results_df)} recipes remaining for final output.")

    # Step 6: Format the final DataFrame into the desired list of dictionaries.
    recommendations, final_message = _format_recommendations(
        final_results_df,
        user_query,
        fallback_msg
    )

    print(f"--- [recipe_pipeline] Complete. Returning {len(recommendations)} recommendations. ---")
    # Return the formatted list and any message (e.g., taste fallback message).
    return recommendations, final_message