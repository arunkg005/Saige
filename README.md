# SAIGE

SAIGE is a Django-based recipe recommendation web app.  
It combines user input, Gemini-based query understanding, and a local ML recommendation pipeline to suggest recipes by ingredients, diet preference, and taste profile.

## Features

- User login/signup flow
- Landing page with recipe categories
- Search/recommendation page for recipe discovery
- AI-assisted parsing of free-text food queries
- ML recommendation pipeline using TF-IDF similarity and taste filtering

## Tech Stack

- Python
- Django
- PostgreSQL
- Pandas / scikit-learn
- Google Generative AI (Gemini)

## Project Structure

- `saige/` - Django project settings and root URL config
- `landing/` - Landing page and category views
- `register/` - Authentication views/templates
- `recc/` - Recommendation UI and ML pipeline
- `static/` - CSS, images, and static assets

## Prerequisites

- Python 3.10+ (recommended)
- PostgreSQL running locally
- pip

## Setup

1. Clone the repository and move into it:
   ```bash
   git clone <repo-url>
   cd Saige
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements_noversion.txt
   ```
4. Configure environment:
   - Create a `.env` file in the project root (or `saige/`).
   - Add your Gemini key:
     ```env
     saige_key=YOUR_GEMINI_API_KEY
     ```
5. Configure PostgreSQL connection in `saige/settings.py` (`NAME`, `USER`, `PASSWORD`, `HOST`, `PORT`).
6. Run migrations:
   ```bash
   python manage.py migrate
   ```
7. Start the development server:
   ```bash
   python manage.py runserver
   ```

Open: `http://127.0.0.1:8000/`

## Useful Commands

- Run tests:
  ```bash
  python manage.py test
  ```
- Create admin user:
  ```bash
  python manage.py createsuperuser
  ```

## Notes

- Recommendation assets are loaded from `recc/saige_model/`:
  - `clustered_recipes.csv`
  - `tfidf_vectorizer.pkl`
  - `tfidf_matrix.pkl`
- Ensure these files are present before using the recommendation flow.
