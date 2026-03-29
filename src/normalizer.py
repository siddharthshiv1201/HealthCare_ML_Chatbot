SYMPTOM_SYNONYMS = {
    # Pain
    "stomach ache": "abdominal pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "tummy pain": "abdominal pain",

    # Vomiting / nausea
    "throwing up": "vomiting",
    "puking": "vomiting",
    "nauseous": "nausea",

    # Fever
    "high temperature": "fever",
    "feverish": "fever",

    # Chills / malaria type
    "chills": "shivering",
    "cold shivers": "shivering",
    "shaking": "shivering",

    # Sweating
    "sweating": "excessive sweating",
    "night sweats": "excessive sweating",

    # Breathing
    "breathlessness": "shortness of breath",
    "difficulty breathing": "shortness of breath",

    # Light sensitivity / migraine
    "light hurts my eyes": "sensitivity to light",
    "bright light hurts": "sensitivity to light",
    "light sensitivity": "sensitivity to light",

    # Headache
    "head pain": "headache",
    "migraine pain": "headache",

    # Skin
    "red spots": "skin rash",
    "skin spots": "skin rash",

    # Yellowing
    "yellow skin": "yellowish skin",
    "yellow eyes": "yellowish eyes",

    # Weakness
    "tired": "fatigue",
    "weakness": "fatigue",
}


def normalize_text(text):
    text = text.lower()

    for phrase, replacement in SYMPTOM_SYNONYMS.items():
        if phrase in text:
            text = text.replace(phrase, replacement)

    return text
