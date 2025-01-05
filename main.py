
import argparse
import pandas as pd
from modules.config import api_token
from modules.utils import classify_text
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def initialize_pipeline():
    # Charger le modèle et le tokenizer
    model_name = "mlabonne/Marcoro14-7B-slerp" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Configurer le modèle avec un prompt template
    template = "Classifie ce texte: {text}"
    prompt = PromptTemplate(template=template, input_variables=["text"])
    return LLMChain(llm=llm, prompt=prompt)

def classify_dataframe(input_path, llm_chain):
    # Charger le fichier CSV
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne nommée 'text'.")
    
    # Appliquer la classification sur chaque ligne
    print("Classification des textes...")
    df["label"] = df["text"].apply(lambda text: classify_text(text))
    
    # Sauvegarder le DataFrame modifié
    df.to_csv("output_labeled.csv", index=False)
    print(f"Résultats enregistrés dans output")

if __name__ == "__main__":
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Classification des textes dans un fichier CSV.")
    parser.add_argument("--input", required=False,  # L'argument n'est plus requis car une valeur par défaut est définie
    default="allianz.csv", help="Chemin du fichier CSV en entrée contenant une colonne 'text'.")
    args = parser.parse_args()
    
    # Initialisation du pipeline
    print("Initialisation du pipeline...")
    llm_chain = initialize_pipeline()
    
    # Exécuter la classification sur le fichier d'entrée
    classify_dataframe(args.input, llm_chain)
