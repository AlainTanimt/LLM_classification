
# LLM Categorisation

Dans une extention du Test_Technique réalisé au répertoire (github.com/AlainTanimt/Test_Technique). Ici il est question ici d'utiliser une approche différente pour réaliser la classification des éléments des rapports SFCR extraits permettant de différencier le contenu utile du contenu indésirable (paragraphe, titre, inutile).

# Motivations

**`Activité et Résultats`**, **`Les entités reportent à leurs Conseils d’Administration respectifs.`** ou **`SCR ( 3 ) MCR ( 4 )`**, à la lecture de ces textes, l'humain peut facilement déterminer s'il s'agit d'un titre, d'un paragraphe ou d'un contenu inutile, uniquement grâce à la forme grammatical du texte. Cette capacité intuitive ne nécessite même pas de considérer des caractéristiques physiques comme la taille ou la position suivant X et Y.
La contrainte ici est l’absence de données étiquetées nécessaires pour effectuer une classification supervisée. Cela exclut non seulement le fine-tuning, mais aussi d’autres approches supervisées comme Random Forests, Recurrent Neural Networks ou k-Nearest Neighbors...
Face à ces limitations, notre démarche explore une solution en tirant parti des capacités des LLM, sans avoir recours à l’étiquetage manuel ou à l’ajustement spécifique des paramètres. Ce projet ambitionne de proposer des méthodes alternatives permettant aux modèles de catégoriser les textes de manière autonome et performante, en surmontant les obstacles liés à l’absence de données structurées.

# Contexte

Les rapports SFCR (« Solvency and Financial Condition Reports ») sont publiés par les compagnies d’assurance pour fournir des informations sur leur situation financière, leur solvabilité et leur gestion des risques. Ces documents contiennent souvent des éléments superflus (comme les bas de pages, hauts de pages, tableaux et graphiques), rendant leur analyse automatique complexe. L'objectif principal est de classifier les éléments (paragraphe, titre, inutile) afin d'extraire uniquement le contenu utile pour faciliter les traitements ultérieurs.


# Description des fichiers et dossiers

- `main.py` : Point d'entrée principal du projet. Initialise le pipeline et exécute la classification sur le fichier CSV d'entrée.
- `data/` : Contient les fichiers CSV d'entrée supplémentaires.
- `modules/` : Contient les modules utilitaires et de configuration.
  - `config.py` : Contient les configurations globales, comme les tokens API.
  - `utils.py` : Contient les fonctions utilitaires, comme la fonction de classification de texte.
- `output/` : Dossier pour les fichiers de sortie générés.
- `README.md` : Ce fichier, contenant la documentation du projet.
- `requirements.txt` : Liste des dépendances Python nécessaires pour exécuter le projet.


# Installation et Exécution
## Prérequis
- Python 3.8+
- Librairies Python : langchain_community, huggingface_hub, transformers, torch, accelerate, pandas

## Instructions
1. Clonez ce repository :
   ```bash
   git clone github.com/AlainTanimt/LLM_classification
   cd <nom-du-repo>
   ```
2. configurez le fichier modules/config.py:
    Inscriver votre token huggingface dans ce fichier afin de permettre de télécharger et d'utiliser les modèles nécessaires à l'exécution du projet.

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Exécutez le script principal :
   ```bash
   python main.py
   ```
Par défaut, le script utilisera le fichier allianz.csv comme entrée (une version courte, extraite du csv complet présent dans le dossier data). Vous pouvez spécifier un autre fichier CSV en utilisant l'argument --input :

# Références 

Ce projet s'inspire notamment des travaux de [Dave Ebbelaar](gist.github.com/daveebbelaar/d65f30bd539a9979d9976af80ec41f07) Son approche a servi de base pour permettre au LLM de faire de la classification sans fine-tuning.

Le modèle de langage large (LLM) utilisé dans ce projet est mlabonne/Marcoro14-7B-slerp, disponible sur Hugging Face. Ce modèle est le résultat d'une fusion innovante de plusieurs modèles développés par MistralAI, affinés à l'aide de techniques de fine-tuning. [Article de Towards Data Science](https://towardsdatascience.com/merge-large-language-models-with-mergekit-2118fb392b54)
