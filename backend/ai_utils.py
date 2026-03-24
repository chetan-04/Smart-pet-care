"""
Lightweight AI-style helper functions used by the SmartPet Care app.

The goal is to keep the logic easy to understand for beginners while still
showing how simple rule-based AI and a pre-trained image model can be used.
"""

import os
from typing import List, Tuple

from PIL import Image

try:
    # TensorFlow is optional but provides a nice way to use a pre-trained model.
    # If it is not installed, the project still runs but image prediction
    # will fall back to a friendly message.
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions,
    )

    _TF_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _TF_AVAILABLE = False


_model = None


def _load_model():
    """
    Lazy-load the MobileNetV2 model.
    The model is loaded the first time we need it and then kept in memory.
    """
    global _model
    if not _TF_AVAILABLE:
        return None
    if _model is None:
        _model = MobileNetV2(weights="imagenet")
    return _model


def predict_pet_species(image_path: str) -> str:
    """
    Use a pre-trained MobileNetV2 model to guess if the image is a dog or cat.

    This uses ImageNet labels and looks for any top prediction that belongs
    to the dog or cat category. For a student project this level of accuracy
    is usually more than enough.
    """
    model = _load_model()
    if model is None:
        return "AI model is not installed. Please install TensorFlow to enable image detection."

    if not os.path.exists(image_path):
        return "Image not found on server."

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = preprocess_input(x)
    x = tf.expand_dims(x, axis=0)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    # ImageNet has many dog/cat breeds; we just check the WordNet IDs.
    dog_keywords = ["dog"]
    cat_keywords = ["cat"]

    for _id, label, score in decoded:
        lower_label = label.lower()
        if any(k in lower_label for k in dog_keywords):
            return f"Detected: Dog (confidence ~{score:.2f})"
        if any(k in lower_label for k in cat_keywords):
            return f"Detected: Cat (confidence ~{score:.2f})"

    # If none of the top predictions is clearly a dog or cat we say unknown.
    best_label = decoded[0][1].replace("_", " ").title()
    return f"Could not confidently detect dog or cat. Closest match: {best_label}"


def suggest_diet(species: str, age_years: float, weight_kg: float) -> str:
    """
    Very simple rule-based "AI" diet suggestion.
    This is implemented with readable if/else logic on purpose.
    """
    species = species.lower()

    if species == "dog":
        if age_years < 1:
            return "Puppy diet: 3–4 small meals per day, high-quality puppy kibble rich in protein and DHA."
        if age_years < 7:
            return "Adult dog diet: 2 balanced meals per day. Choose food for active adult dogs and avoid overfeeding."
        return "Senior dog diet: easy-to-digest food, slightly lower calories, and joint-support supplements if vet recommends."

    if species == "cat":
        if age_years < 1:
            return "Kitten diet: 3–4 small meals of kitten-specific food, rich in protein and fat."
        if age_years < 7:
            return "Adult cat diet: 2–3 meals per day. Use complete dry or wet food and encourage water intake."
        return "Senior cat diet: controlled calories, kidney-friendly food if advised by vet, and regular health checks."

    if species == "horse":
        if age_years < 3:
            return "Young horse diet: mostly good-quality forage with carefully introduced concentrates; avoid overfeeding to protect joints."
        return "Adult horse diet: constant access to hay or pasture, small grain meals if needed, and plenty of clean water and salt."

    if species == "rabbit":
        return "Rabbit diet: unlimited grass hay, a small amount of high-fiber pellets, and fresh leafy greens; avoid sugary treats."

    if species == "parrot":
        return "Parrot diet: a mix of quality pellets, fresh vegetables, some fruit, and limited seeds; avoid chocolate, avocado, and caffeine."

    if species == "hamster":
        return "Hamster diet: commercial hamster mix with grains and seeds, plus small pieces of vegetables and occasional protein treats."

    if species == "turtle":
        return "Turtle diet: species-dependent mix of commercial turtle pellets, leafy greens, and occasional insects or fish; provide calcium."

    if species == "fish":
        return "Fish diet: small portions of quality fish flakes or pellets once or twice a day; avoid overfeeding to keep water clean."

    # Generic fallback if species is unknown
    return "Provide clean water at all times and a balanced commercial diet appropriate for your pet's species and size."


def get_vaccination_schedule(species: str, age_years: float) -> List[str]:
    """
    Return a simple text-based vaccination schedule.
    Real schedules depend on location; this is only for educational use.
    """
    species = species.lower()
    schedule: List[str] = []

    if species == "dog":
        schedule.append("Core: Distemper, Parvovirus, Adenovirus, Rabies.")
        if age_years < 1:
            schedule.append("Puppies: multiple booster shots every 3–4 weeks until 16 weeks of age.")
        else:
            schedule.append("Adults: booster every 1–3 years depending on vet recommendation.")
        schedule.append("Optional: Kennel cough, Leptospirosis, depending on lifestyle and region.")
    elif species == "cat":
        schedule.append("Core: Feline panleukopenia, herpesvirus, calicivirus, Rabies.")
        if age_years < 1:
            schedule.append("Kittens: series of vaccines every 3–4 weeks until about 16 weeks.")
        else:
            schedule.append("Adults: booster every 1–3 years as advised by your vet.")
        schedule.append("Optional: FeLV (feline leukemia) for outdoor or high-risk cats.")
    elif species == "horse":
        schedule.append("Core: Tetanus, Equine influenza, Equine herpesvirus, and Rabies (where recommended).")
        if age_years < 1:
            schedule.append("Foals: series of vaccines beginning at a few months of age, with boosters as advised by a vet.")
        else:
            schedule.append("Adults: boosters usually every 6–12 months depending on disease risk and travel.")
    elif species == "rabbit":
        schedule.append("In some regions, vaccines are available for myxomatosis and rabbit haemorrhagic disease (RHD).")
        schedule.append("Ask your local vet which vaccines are recommended for rabbits in your area.")
    elif species == "parrot":
        schedule.append("Parrots often rely more on good hygiene and quarantine than routine vaccines.")
        schedule.append("Your vet may recommend specific tests or vaccines based on species and local diseases.")
    elif species == "hamster":
        schedule.append("Hamsters generally do not receive routine vaccinations.")
        schedule.append("Focus on clean housing, good diet, and prompt vet visits if illness is suspected.")
    elif species == "turtle":
        schedule.append("Turtles do not usually receive routine vaccinations.")
        schedule.append("Clean water, correct temperature, UVB lighting, and proper diet are the main health protections.")
    elif species == "fish":
        schedule.append("Fish rarely receive individual vaccinations in a home aquarium.")
        schedule.append("Good water quality, filtration, and quarantine of new fish are the best disease prevention tools.")
    else:
        schedule.append("Vaccination needs depend heavily on species. Please consult a veterinarian.")

    return schedule


def analyze_symptoms(species: str, symptoms: List[str]) -> Tuple[str, str]:
    """
    Rule-based symptom checker.

    Returns:
        severity: short label (e.g. 'Mild', 'Moderate', 'Emergency')
        advice: user-friendly description with a vet recommendation.
    """
    species = species.lower()
    selected = {s.lower() for s in symptoms}

    # Assign simple scores to each symptom to calculate severity.
    scores = {
        "fever": 2,
        "vomiting": 3,
        "diarrhea": 2,
        "loss_of_appetite": 2,
        "weakness": 3,
        "coughing": 2,
        "sneezing": 1,
        "itching": 1,
        "difficulty_breathing": 5,
        "seizures": 6,
    }

    total_score = sum(scores.get(symptom, 1) for symptom in selected)

    if total_score == 0:
        return (
            "Unknown",
            "No symptoms selected. Please choose at least one symptom so we can give basic advice.",
        )

    if total_score <= 3:
        severity = "Mild"
        advice = (
            "Your pet seems to have mild symptoms. Monitor closely, ensure fresh water, "
            "and keep them comfortable. If symptoms last more than 24 hours, contact a vet."
        )
    elif total_score <= 7:
        severity = "Moderate"
        advice = (
            "Your pet may be experiencing a moderate health issue. Limit activity, offer small amounts "
            "of water, and book a routine vet visit within the next 24 hours."
        )
    else:
        severity = "Emergency"
        advice = (
            "Your pet may be in a serious condition. Seek emergency veterinary help immediately. "
            "Do not wait if you notice difficulty breathing, seizures, or your pet collapses."
        )

    # Add a short species-specific tail to the message so it feels more personalized.
    if species == "dog":
        advice += " For dogs, avoid giving human painkillers without explicit veterinary guidance."
    elif species == "cat":
        advice += " For cats, never give paracetamol or ibuprofen as they can be highly toxic."

    return severity, advice

