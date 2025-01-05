from enum import Enum
from pydantic import BaseModel

class Label(str, Enum):
    TITRE = "Titre"
    PARAGRAPHE = "Paragraphe"
    INUTILE = "Inutile"

class LabelClassification(BaseModel):
    category: Label


def extraction_category(texte: str) -> str:
    # Vérifie si "Assistant:" est présent dans le texte
    if "Assistant:" in texte:
        # Divise le texte à partir de "Assistant:" et récupère le mot suivant
        mot_suivant = texte.split("Assistant:")[1].strip().split()[0]
        # Enlève le point (.) à la fin du mot, s'il y en a un
        if mot_suivant.endswith('.'):
            mot_suivant = mot_suivant[:-1]
        return mot_suivant
    else:
        return None

def classify_text(text: str) -> LabelClassification:
    # Obtenir la sortie brute du modèle
    raw_output = llm_chain.run(text=text)

    try:
        category = extraction_category(raw_output)
        if category not in Label.__members__.values():
            raise ValueError(f"Label non valide : {category}")
        return LabelClassification(category=Label(category))
    except Exception as e:
        raise ValueError(f"Erreur dans le format de sortie du modèle: {category}") from e
