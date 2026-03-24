# SmartPet Care – AI Based Pet Health and Care Assistant

SmartPet Care is a beginner-friendly **full-stack web application** that helps pet owners manage
their pets' basic health information, diet suggestions, vaccination guidance, and simple AI tools
like symptom checking and image-based pet type detection.

The project is built as a **final year project** using:

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python **Flask**
- **Database**: SQLite (via `flask_sqlalchemy`)
- **AI Features**:
  - Rule-based diet and vaccination suggestions
  - Rule-based symptom checker
  - Optional pre-trained image classifier (MobileNetV2 via TensorFlow) for dog vs cat detection
  - AI health detection (image-based) using OpenCV preprocessing + lightweight scoring (stores history)
  - Smart meal calculator and diet plan engine (species + age + weight)
  - Hygiene & care module with reminders

The code aims to be **easy to read and run locally**.

---

## Project structure

```text
animalcare/
├─ backend/
│  ├─ app.py                # Main Flask application (routes, models, auth)
│  ├─ ai_utils.py           # Simple AI helper functions and image detection
│  ├─ health_ai.py           # Image-based health detection (OpenCV + lightweight AI scoring)
│  ├─ smart_care.py          # Meal calculator, diet plans, hygiene tips, smart suggestions
│  ├─ templates/            # Jinja2 HTML templates used by Flask
│  │  ├─ base.html
│  │  ├─ index.html
│  │  ├─ login.html
│  │  ├─ register.html
│  │  ├─ dashboard.html
│  │  ├─ add_pet.html
│  │  ├─ pet_detail.html
│  │  ├─ care_guide.html
│  │  ├─ symptom_checker.html
│  │  ├─ image_detect.html
│  │  ├─ health_detect.html
│  │  ├─ health_history.html
│  │  ├─ meal_recommendation.html
│  │  ├─ diet_recommendation.html
│  │  └─ hygiene_tips.html
│  └─ static/
│     ├─ css/
│     │  └─ style.css       # Modern responsive styling
│     └─ js/
│        └─ main.js         # Small JS helpers (nav toggle, etc.)
│
├─ frontend/
│  └─ index.html            # Standalone static demo of the UI layout
│
├─ database/
│  └─ smartpetcare.db       # SQLite database (auto-created on first run)
│
├─ requirements.txt         # Python dependencies
└─ README.md                # This file
```

> Note: The SQLite file `smartpetcare.db` is created automatically inside the `database` folder
> the first time you run the Flask server, so you do not need to create it manually.

---

## Features overview

- **User authentication**
  - Registration and login with username/email and password
  - Passwords are stored as secure hashes (using Werkzeug)
  - Session-based authentication using `flask_login`

- **Pet profile management**
  - Add pet details (name, type: dog/cat, age in years, weight in kg)
  - Each pet is linked to a specific user
  - Dashboard view showing all pets for the logged-in user

- **Pet care guide**
  - Dedicated page with:
    - Dog care tips
    - Cat care tips
    - Diet guidance by life stage (puppy/kitten, adult, senior)
    - General vaccination overview

- **Symptom checker (simple AI rules)**
  - User selects one of their pets
  - User checks symptoms (fever, vomiting, weakness, etc.)
  - The system computes a simple severity score and returns:
    - **Severity label**: Mild / Moderate / Emergency
    - **Advice text** and **recommendation to visit a vet**

- **AI image detection (pre-trained model)**
  - User uploads an image of a pet
  - Backend uses **MobileNetV2** (ImageNet pre-trained model via TensorFlow) to:
    - Classify the image
    - Check whether the predicted class is a dog or a cat
    - Show the result, or a friendly message if it cannot be determined
  - If TensorFlow is not installed, the app will show a message that the AI model is unavailable,
    but the rest of the site still works.

- **AI health detection (image-based)**
  - Upload a pet image to detect:
    - **Healthy**
    - **Sick / possible health issues**
  - If sick, suggests one of:
    - skin problems
    - infection signs
    - weakness indicators
  - Shows a confidence percentage
  - Stores a history of health checks in SQLite

- **Weight-based meal recommendation**
  - Input pet type, age, weight
  - Shows daily food amount (grams), meals per day, and food type suggestions

- **Diet recommendation engine**
  - Generates healthy food list + avoid list based on species, age, and weight

- **Hygiene and care module**
  - Grooming, bathing, cleaning tips
  - Vaccination reminders and general health maintenance

- **Dashboard**
  - Shows cards for each pet with:
    - Basic profile info
    - AI-based diet suggestion
    - Bullet list of vaccination schedule notes

- **UI / UX**
  - Modern layout with:
    - Sticky navigation bar
    - Cards, soft shadows, simple gradients
    - Responsive grid layout
  - Mobile navigation toggle implemented with JavaScript

---

## How to run the project locally

### 1. Install Python

Make sure **Python 3.9+** is installed and added to your `PATH`.

You can check by running:

```bash
python --version
```

On some systems you may need to use:

```bash
py --version
```

### 2. Create and activate a virtual environment (recommended)

From the project root (`animalcare`):

```bash
python -m venv venv
```

Activate it on **Windows (PowerShell)**:

```bash
venv\Scripts\Activate.ps1
```

On **Windows (cmd)**:

```bash
venv\Scripts\activate.bat
```

On **Linux / macOS**:

```bash
source venv/bin/activate
```

### 3. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

> Notes:
> - `tensorflow` is optional (only needed for **Dog vs Cat** detection on `/image-detect`).
> - The **AI health detection** feature runs on CPU using `numpy` + `pillow`.
> - If you want faster preprocessing, you can optionally install OpenCV:
>   - `pip install opencv-python-headless`

### 4. Run the Flask application

Change into the `backend` folder and run the app:

```bash
cd backend
python app.py
```

You should see output similar to:

```text
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Open a browser and go to:

- `http://127.0.0.1:5000/`

The database and tables will be **created automatically** on first request.

---

## New pages (after upgrade)

After login, use the navigation menu:

- `Health Detection` → `/health-detect`
- `Health History` → `/health-history`
- `Meal Plan` → `/meal-recommendation`
- `Diet Plan` → `/diet-recommendation`
- `Hygiene Tips` → `/hygiene-tips`

---

## New API routes (JSON)

All APIs require login (session cookie):

- `POST /api/predict-health` (multipart form-data: `image`, optional `pet_id`)
- `POST /api/meal-calc` (JSON: `{ "species": "...", "age_years": 2, "weight_kg": 10 }`)
- `POST /api/diet-plan` (JSON: `{ "species": "...", "age_years": 2, "weight_kg": 10 }`)
- `POST /api/health-suggestions` (JSON: `{ "condition": "Sick", "issue": "skin problems" }`)

---

## Usage walkthrough

1. **Register a new user**
   - Click **Get Started** or **Register**
   - Fill in username, email, and password
   - After registering, login with your credentials

2. **Add a pet profile**
   - Go to the **Dashboard**
   - Click **Add new pet**
   - Enter name, type (dog/cat), age, and weight

3. **View dashboard**
   - See all your pets as cards
   - Each card shows:
     - Age and weight
     - AI-based diet suggestion
     - Short vaccination notes

4. **Use the care guide**
   - Open the **Care Guide** from the navigation menu
   - Review dog and cat tips, diet by age, and vaccination overview

5. **Use the symptom checker**
   - Open **Symptom Checker**
   - Select one of your pets
   - Check the symptoms you observe
   - Submit the form to view:
     - Severity label
     - Text advice and recommendation

6. **Use AI image detection**
   - Open **AI Image Detect**
   - Upload a picture of a dog or a cat
   - The app will try to guess which one it is using the pre-trained model

---

## Important notes (for report / viva)

- **Security**
  - This is a demo app for learning:
    - Passwords are hashed, but no password reset is implemented.
    - There is a single `SECRET_KEY` set in the code; in production it should be an environment variable.

- **AI limitations**
  - Symptom checker and diet/vaccination suggestions are **rule-based** and simplified.
  - Image detection uses a general-purpose ImageNet model, not a dedicated vet model.
  - This system **does not replace professional veterinary advice**.

- **Where to explain AI in your report**
  - `backend/ai_utils.py`:
    - `suggest_diet` – age and species-based diet rules.
    - `get_vaccination_schedule` – species-based vaccination notes.
    - `analyze_symptoms` – scoring system converting symptoms to severity.
    - `predict_pet_species` – pre-trained MobileNetV2 image classification.

---

## Beginner tips

- To see how routes work, start from `backend/app.py`:
  - Look for `@app.route("/")`, `@app.route("/dashboard")`, etc.
- To customize the look and feel:
  - Edit `backend/static/css/style.css`.
- To change messages or texts:
  - Edit the templates in `backend/templates/`.

You can extend this project by:

- Adding more pet types (rabbits, birds, etc.)
- Adding appointment reminders
- Integrating an external veterinary API (for clinics or medicines)
- Building charts for weight or age over time

