"""
Smart care engine (meal, diet, hygiene, and suggestions).

This module is deliberately rule-based and beginner-friendly, but exposes
clear functions that can be used by web routes and APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


def _age_group(age_years: float) -> str:
    if age_years < 1:
        return "young"
    if age_years < 7:
        return "adult"
    return "old"


@dataclass(frozen=True)
class MealRecommendation:
    daily_food_grams: int
    meals_per_day: int
    food_type_suggestions: List[str]
    notes: List[str]


def calculate_meal_plan(species: str, age_years: float, weight_kg: float) -> MealRecommendation:
    """
    Weight-based daily meal calculator (simple educational logic).

    We approximate daily food (grams) using a species multiplier and weight.
    Real needs depend on activity, breed, health, and vet advice.
    """
    species = (species or "").lower().strip()
    age = _age_group(age_years)

    # Species multipliers (grams per kg per day).
    multipliers = {
        "dog": 30,
        "cat": 35,
        "rabbit": 40,
        "horse": 20,
        "parrot": 12,
        "hamster": 10,
        "turtle": 8,
        "fish": 6,
    }
    mult = multipliers.get(species, 28)

    # Age adjustment.
    if age == "young":
        mult *= 1.15
        meals = 3
    elif age == "old":
        mult *= 0.9
        meals = 2
    else:
        meals = 2

    daily = max(int(round(mult * max(weight_kg, 0.2))), 10)

    food_types = []
    notes = []

    if species in {"dog", "cat"}:
        food_types = [
            "High-quality protein",
            "Balanced dry food (kibble)",
            "Natural food (vet-approved home diet)",
        ]
        if species == "cat":
            notes.append("Cats benefit from more moisture (wet food) and fresh water availability.")
    elif species == "rabbit":
        food_types = ["Unlimited hay", "Leafy greens", "High-fiber pellets (small portion)"]
        notes.append("Avoid sugary fruits and starchy treats.")
    elif species == "horse":
        food_types = ["Forage (hay/pasture)", "Mineral/salt block", "Concentrates if needed"]
        notes.append("Most calories should come from forage; split concentrates into small meals.")
    else:
        food_types = ["Species-appropriate commercial diet", "Fresh water", "Occasional natural foods (safe list)"]

    return MealRecommendation(
        daily_food_grams=daily,
        meals_per_day=meals,
        food_type_suggestions=food_types,
        notes=notes,
    )


@dataclass(frozen=True)
class DietPlan:
    title: str
    healthy_foods: List[str]
    avoid_foods: List[str]
    tips: List[str]


def build_diet_plan(species: str, age_years: float, weight_kg: float) -> DietPlan:
    species = (species or "").lower().strip()
    age = _age_group(age_years)

    base_tips = [
        "Introduce any new food slowly over 5–7 days.",
        "Keep fresh water available at all times.",
        "If your pet has allergies or medical conditions, follow your vet’s diet plan.",
    ]

    if species == "dog":
        healthy = ["Lean meats", "Complete dog food", "Cooked rice/oats (small amounts)", "Carrots/pumpkin (small amounts)"]
        avoid = ["Chocolate", "Grapes/raisins", "Onions/garlic", "Cooked bones", "Xylitol (sweetener)"]
        tips = base_tips + (["Young dogs need higher calories and 3–4 smaller meals."] if age == "young" else [])
        return DietPlan("Dog diet plan", healthy, avoid, tips)

    if species == "cat":
        healthy = ["High-protein cat food", "Wet food (moisture)", "Cooked chicken/fish (plain)", "Cat-safe treats (small)"]
        avoid = ["Paracetamol/ibuprofen (toxic)", "Onions/garlic", "Chocolate", "Milk (many cats are lactose intolerant)"]
        tips = base_tips + ["Cats are obligate carnivores; prioritize animal protein."]
        return DietPlan("Cat diet plan", healthy, avoid, tips)

    if species == "rabbit":
        healthy = ["Unlimited grass hay", "Leafy greens", "High-fiber pellets (small portion)"]
        avoid = ["Iceberg lettuce", "Sugary snacks", "Bread/cereal", "Too many fruits"]
        tips = base_tips + ["Hay is the most important part of a rabbit’s diet."]
        return DietPlan("Rabbit diet plan", healthy, avoid, tips)

    if species == "horse":
        healthy = ["Hay/pasture", "Beet pulp (if advised)", "Mineral supplements (vet/farrier advice)"]
        avoid = ["Sudden diet changes", "Moldy hay", "Excess grain (risk of colic/laminitis)"]
        tips = base_tips + ["Split concentrates into multiple small meals."]
        return DietPlan("Horse diet plan", healthy, avoid, tips)

    if species in {"parrot", "bird"}:
        healthy = ["Quality pellets", "Fresh vegetables", "Some fruit (small)", "Limited seeds"]
        avoid = ["Avocado", "Chocolate", "Caffeine", "Alcohol"]
        tips = base_tips + ["Avoid non-stick fumes and aerosols around birds."]
        return DietPlan("Bird diet plan", healthy, avoid, tips)

    # Generic fallback
    healthy = ["Species-appropriate commercial diet", "Fresh vegetables (safe list)", "Clean water"]
    avoid = ["Highly salty foods", "Highly sugary foods", "Unknown human medicines"]
    tips = base_tips + ["Search a vet-approved safe food list for your exact species."]
    return DietPlan(f"{species.title() if species else 'Pet'} diet plan", healthy, avoid, tips)


def health_suggestions(condition: str, issue: Optional[str]) -> Dict[str, str]:
    """
    Convert detection output into user-facing guidance.
    """
    condition = (condition or "").lower().strip()
    issue = (issue or "").lower().strip() if issue else None

    if condition == "healthy":
        return {
            "possible_issue": "No major issue detected",
            "recommended_action": "Continue normal care",
            "basic_treatment": "Maintain hygiene, balanced diet, and routine vet checkups.",
        }

    if issue == "skin problems":
        return {
            "possible_issue": "Skin irritation / rash / parasite possibility",
            "recommended_action": "Home care + vet if it persists",
            "basic_treatment": "Check for fleas/ticks, avoid harsh shampoos, keep area clean and dry. Visit a vet if redness spreads or itching is severe.",
        }
    if issue == "infection signs":
        return {
            "possible_issue": "Possible infection signs",
            "recommended_action": "Visit vet",
            "basic_treatment": "Monitor temperature and appetite. Keep the pet hydrated and comfortable. Seek veterinary advice promptly for discharge, swelling, or fever.",
        }
    if issue == "weakness indicators":
        return {
            "possible_issue": "Low energy / weakness indicators",
            "recommended_action": "Monitor closely; visit vet if worsening",
            "basic_treatment": "Ensure water, small frequent meals, and rest. If your pet collapses, refuses food, or has breathing difficulty, seek emergency care.",
        }

    return {
        "possible_issue": "Possible health issue",
        "recommended_action": "Monitor and consult vet",
        "basic_treatment": "This is a basic AI-based warning only. If you’re concerned, contact a veterinarian.",
    }


def hygiene_tips(species: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Hygiene and care module content.
    Returns sections: grooming, bathing, cleaning, reminders.
    """
    species = (species or "").lower().strip()

    grooming = [
        "Brush coat regularly to remove dirt and check for skin changes.",
        "Trim nails safely or ask a groomer/vet for help.",
        "Check ears and eyes for unusual discharge or smell.",
    ]
    bathing = [
        "Use pet-safe shampoo only; avoid human products.",
        "Dry thoroughly after bathing to prevent fungal growth.",
    ]
    cleaning = [
        "Wash food/water bowls daily and keep bedding clean and dry.",
        "Keep living area ventilated and free from sharp objects.",
    ]
    reminders = [
        "Keep vaccinations up to date based on vet schedule.",
        "Deworming and parasite prevention should follow local vet advice.",
        "Schedule routine vet checkups at least once a year (twice for seniors).",
    ]

    if species in {"cat"}:
        bathing = ["Most cats don’t need frequent baths; focus on brushing and spot cleaning."]
    if species in {"rabbit"}:
        bathing = ["Do not fully bathe rabbits unless advised by a vet; use gentle spot cleaning."]
    if species in {"parrot", "bird"}:
        cleaning.append("Clean cages and perches frequently; avoid aerosol cleaners near birds.")

    return {
        "grooming": grooming,
        "bathing": bathing,
        "cleaning": cleaning,
        "reminders": reminders,
    }

