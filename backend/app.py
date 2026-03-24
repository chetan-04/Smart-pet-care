import os
import json
from datetime import datetime

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    send_from_directory,
    jsonify,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user,
    UserMixin,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from ai_utils import analyze_symptoms, suggest_diet, get_vaccination_schedule, predict_pet_species
from health_ai import predict_health_from_image
from smart_care import calculate_meal_plan, build_diet_plan, health_suggestions, hygiene_tips


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "database", "smartpetcare.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")


def create_app():
    """
    Application factory that configures Flask, database, and login manager.
    This keeps the setup beginner friendly but still organized.
    """
    app = Flask(__name__)

    # Basic secret key for sessions (for demo purposes only)
    app.config["SECRET_KEY"] = "change-this-in-production"

    # SQLite configuration
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # File upload configuration for pet images
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    return app


app = create_app()
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# Central list of supported pet types. This keeps labels and icons
# in one place so templates and views stay consistent.
PET_TYPES = {
    "dog": {"label": "Dog", "icon": "🐶"},
    "cat": {"label": "Cat", "icon": "🐱"},
    "horse": {"label": "Horse", "icon": "🐴"},
    "rabbit": {"label": "Rabbit", "icon": "🐰"},
    "parrot": {"label": "Parrot / Bird", "icon": "🦜"},
    "hamster": {"label": "Hamster", "icon": "🐹"},
    "turtle": {"label": "Turtle", "icon": "🐢"},
    "fish": {"label": "Fish", "icon": "🐟"},
}


class User(UserMixin, db.Model):
    """
    Simple user model for authentication.
    Passwords are stored as hashes using Werkzeug helpers.
    """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    pets = db.relationship("Pet", backref="owner", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Pet(db.Model):
    """
    Pet profile model.
    Stores basic information needed for care suggestions and symptom checks.
    """

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    # Species string (e.g. "dog", "cat", "horse", "rabbit", etc.)
    # Existing databases will continue to work; new species are just new string values.
    species = db.Column(db.String(20), nullable=False)
    age_years = db.Column(db.Float, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class HealthCheck(db.Model):
    """
    Stores image-based health detection history for a user (optionally linked to a pet).
    This supports the "history of health checks" requirement.
    """

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    pet_id = db.Column(db.Integer, db.ForeignKey("pet.id"), nullable=True, index=True)

    image_filename = db.Column(db.String(255), nullable=False)
    condition = db.Column(db.String(20), nullable=False)  # Healthy / Sick
    issue = db.Column(db.String(50), nullable=True)  # skin problems / infection signs / weakness indicators
    confidence = db.Column(db.Float, nullable=False)  # 0..1
    signals_json = db.Column(db.Text, nullable=True)  # optional debugging / transparency


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            flash("Username or email already exists.", "danger")
            return redirect(url_for("register"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username_or_email = request.form.get("username_or_email")
        password = request.form.get("password")

        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully.", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))

        flash("Invalid credentials. Please try again.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    recent_checks = (
        HealthCheck.query.filter_by(owner_id=current_user.id)
        .order_by(HealthCheck.created_at.desc())
        .limit(5)
        .all()
    )

    # Prepare lightweight AI-based suggestions for each pet
    pet_cards = []
    for pet in pets:
         # Resolve human-friendly label and icon for each pet type
        species_key = (pet.species or "").lower()
        species_meta = PET_TYPES.get(species_key, {"label": pet.species.title(), "icon": "🐾"})

        pet_cards.append(
            {
                "pet": pet,
                "species_label": species_meta["label"],
                "icon": species_meta["icon"],
                "diet": suggest_diet(pet.species, pet.age_years, pet.weight_kg),
                "vaccinations": get_vaccination_schedule(pet.species, pet.age_years),
            }
        )

    return render_template("dashboard.html", pet_cards=pet_cards, recent_checks=recent_checks)


@app.route("/pet/add", methods=["GET", "POST"])
@login_required
def add_pet():
    if request.method == "POST":
        name = request.form.get("name")
        species = request.form.get("species")
        age_years = request.form.get("age_years")
        weight_kg = request.form.get("weight_kg")

        if not name or not species or not age_years or not weight_kg:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("add_pet"))

        try:
            age_years = float(age_years)
            weight_kg = float(weight_kg)
        except ValueError:
            flash("Age and weight must be numbers.", "danger")
            return redirect(url_for("add_pet"))

        pet = Pet(
            name=name,
            species=species,
            age_years=age_years,
            weight_kg=weight_kg,
            owner_id=current_user.id,
        )
        db.session.add(pet)
        db.session.commit()
        flash("Pet added successfully.", "success")
        return redirect(url_for("dashboard"))

    # Pass supported pet types so the template can build the dropdown with icons.
    return render_template("add_pet.html", pet_types=PET_TYPES)


@app.route("/pet/<int:pet_id>")
@login_required
def pet_detail(pet_id):
    pet = Pet.query.filter_by(id=pet_id, owner_id=current_user.id).first_or_404()

    diet = suggest_diet(pet.species, pet.age_years, pet.weight_kg)
    vaccinations = get_vaccination_schedule(pet.species, pet.age_years)

    return render_template(
        "pet_detail.html",
        pet=pet,
        diet=diet,
        vaccinations=vaccinations,
    )


@app.route("/care-guide")
@login_required
def care_guide():
    """
    Static-style care guide rendered from template.
    Uses basic, friendly language suitable for beginners.
    """
    return render_template("care_guide.html")


@app.route("/symptom-checker", methods=["GET", "POST"])
@login_required
def symptom_checker():
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    advice = None
    severity = None

    if request.method == "POST":
        pet_id = request.form.get("pet_id")
        selected_symptoms = request.form.getlist("symptoms")

        pet = Pet.query.filter_by(id=pet_id, owner_id=current_user.id).first()
        if not pet:
            flash("Please select a valid pet.", "danger")
        else:
            severity, advice = analyze_symptoms(pet.species, selected_symptoms)

    return render_template(
        "symptom_checker.html",
        pets=pets,
        advice=advice,
        severity=severity,
    )


@app.route("/image-detect", methods=["GET", "POST"])
@login_required
def image_detect():
    prediction = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Please upload an image file.", "danger")
            return redirect(url_for("image_detect"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            prediction = predict_pet_species(save_path)
        except Exception as exc:  # pragma: no cover - defensive
            # If anything goes wrong with the AI model we show a user friendly message.
            error = f"Could not run AI model: {exc}"

    return render_template("image_detect.html", prediction=prediction, error=error)


@app.route("/health-detect", methods=["GET", "POST"])
@login_required
def health_detect():
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    result = None
    suggestions = None
    error = None
    uploaded_url = None

    if request.method == "POST":
        file = request.files.get("image")
        pet_id = request.form.get("pet_id") or None

        if not file or file.filename == "":
            flash("Please upload an image file.", "danger")
            return redirect(url_for("health_detect"))

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        stored_name = f"health_{current_user.id}_{timestamp}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(save_path)
        uploaded_url = url_for("uploaded_file", filename=stored_name)

        try:
            pred = predict_health_from_image(save_path)
            result = {
                "condition": pred.condition,
                "issue": pred.issue,
                "confidence": pred.confidence,
                "confidence_pct": round(pred.confidence * 100, 1),
            }
            suggestions = health_suggestions(pred.condition, pred.issue)

            pet = None
            if pet_id:
                pet = Pet.query.filter_by(id=pet_id, owner_id=current_user.id).first()

            check = HealthCheck(
                owner_id=current_user.id,
                pet_id=pet.id if pet else None,
                image_filename=stored_name,
                condition=pred.condition,
                issue=pred.issue,
                confidence=float(pred.confidence),
                signals_json=json.dumps(pred.signals, ensure_ascii=False),
            )
            db.session.add(check)
            db.session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            error = f"Could not analyze image: {exc}"

    return render_template(
        "health_detect.html",
        pets=pets,
        result=result,
        suggestions=suggestions,
        error=error,
        uploaded_url=uploaded_url,
    )


@app.route("/health-history")
@login_required
def health_history():
    checks = (
        HealthCheck.query.filter_by(owner_id=current_user.id)
        .order_by(HealthCheck.created_at.desc())
        .limit(50)
        .all()
    )
    pet_map = {p.id: p for p in Pet.query.filter_by(owner_id=current_user.id).all()}
    return render_template("health_history.html", checks=checks, pet_map=pet_map)


@app.route("/meal-recommendation", methods=["GET", "POST"])
@login_required
def meal_recommendation():
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    meal = None
    error = None

    if request.method == "POST":
        species = request.form.get("species") or ""
        age_years = request.form.get("age_years") or ""
        weight_kg = request.form.get("weight_kg") or ""

        try:
            meal = calculate_meal_plan(species, float(age_years), float(weight_kg))
        except Exception:
            error = "Please enter valid species, age, and weight values."

    return render_template("meal_recommendation.html", pets=pets, meal=meal, error=error, pet_types=PET_TYPES)


@app.route("/diet-recommendation", methods=["GET", "POST"])
@login_required
def diet_recommendation():
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    plan = None
    error = None

    if request.method == "POST":
        species = request.form.get("species") or ""
        age_years = request.form.get("age_years") or ""
        weight_kg = request.form.get("weight_kg") or ""
        try:
            plan = build_diet_plan(species, float(age_years), float(weight_kg))
        except Exception:
            error = "Please enter valid species, age, and weight values."

    return render_template("diet_recommendation.html", pets=pets, plan=plan, error=error, pet_types=PET_TYPES)


@app.route("/hygiene-tips")
@login_required
def hygiene_module():
    tips = hygiene_tips()
    return render_template("hygiene_tips.html", tips=tips)


# ----------------------
# JSON API routes (new)
# ----------------------

@app.route("/api/predict-health", methods=["POST"])
@login_required
def api_predict_health():
    file = request.files.get("image")
    pet_id = request.form.get("pet_id") or None

    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No image uploaded."}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    stored_name = f"health_{current_user.id}_{timestamp}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(save_path)

    pred = predict_health_from_image(save_path)
    sugg = health_suggestions(pred.condition, pred.issue)

    pet = None
    if pet_id:
        pet = Pet.query.filter_by(id=pet_id, owner_id=current_user.id).first()

    check = HealthCheck(
        owner_id=current_user.id,
        pet_id=pet.id if pet else None,
        image_filename=stored_name,
        condition=pred.condition,
        issue=pred.issue,
        confidence=float(pred.confidence),
        signals_json=json.dumps(pred.signals, ensure_ascii=False),
    )
    db.session.add(check)
    db.session.commit()

    return jsonify(
        {
            "ok": True,
            "result": {
                "condition": pred.condition,
                "issue": pred.issue,
                "confidence": float(pred.confidence),
                "confidence_pct": round(float(pred.confidence) * 100, 1),
            },
            "suggestions": sugg,
            "upload_url": url_for("uploaded_file", filename=stored_name),
            "history_id": check.id,
        }
    )


@app.route("/api/meal-calc", methods=["POST"])
@login_required
def api_meal_calc():
    data = request.get_json(silent=True) or {}
    species = data.get("species", "")
    age_years = data.get("age_years", 0)
    weight_kg = data.get("weight_kg", 0)

    try:
        meal = calculate_meal_plan(species, float(age_years), float(weight_kg))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid inputs."}), 400

    return jsonify(
        {
            "ok": True,
            "meal": {
                "daily_food_grams": meal.daily_food_grams,
                "meals_per_day": meal.meals_per_day,
                "food_type_suggestions": meal.food_type_suggestions,
                "notes": meal.notes,
            },
        }
    )


@app.route("/api/health-suggestions", methods=["POST"])
@login_required
def api_health_suggestions():
    data = request.get_json(silent=True) or {}
    condition = data.get("condition")
    issue = data.get("issue")
    return jsonify({"ok": True, "suggestions": health_suggestions(condition, issue)})


@app.route("/api/diet-plan", methods=["POST"])
@login_required
def api_diet_plan():
    data = request.get_json(silent=True) or {}
    species = data.get("species", "")
    age_years = data.get("age_years", 0)
    weight_kg = data.get("weight_kg", 0)

    try:
        plan = build_diet_plan(species, float(age_years), float(weight_kg))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid inputs."}), 400

    return jsonify(
        {
            "ok": True,
            "plan": {
                "title": plan.title,
                "healthy_foods": plan.healthy_foods,
                "avoid_foods": plan.avoid_foods,
                "tips": plan.tips,
            },
        }
    )

@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    """
    Simple helper to serve uploaded images during development.
    In production you would typically configure your web server directly.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # Ensure the database and tables are created before the first request.
    with app.app_context():
        db.create_all()

    # Running with debug=True is useful for final year projects and demos.
    app.run(debug=True)


