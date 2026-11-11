# Assistant IA pour l'analyse de documents et d'appels d'offres

Projet Python expérimental visant à extraire le contenu de documents PDF, analyser des clauses administratives et exploiter un modèle de langage pour structurer des informations ou assister le remplissage de documents.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `code.py` | Interface Gradio et pipeline d'analyse de documents |
| `cahier.py` | Variante du pipeline orientée cahier des charges et annexes |
| `code2.py` | Prototype de classification de clauses avec Transformers |
| `code3.py` | Prototype de pipeline PDF/OCR/LLM pour le remplissage d'annexes |
| `extraction.py` | API Flask d'extraction et de structuration de CV |
| `try.py` | Prototype Flask d'extraction de texte PDF |

> Ces fichiers correspondent à plusieurs versions et expérimentations du projet. Ils ne constituent pas encore une application unifiée. Vérifiez les chemins et les dépendances avant exécution.

## Prérequis

- Python 3.10 ou version ultérieure (à adapter selon les bibliothèques installées)
- Tesseract OCR avec la langue française (`fra`)
- Poppler pour le traitement des PDF via `pdf2image`
- Une clé API OpenAI pour les fonctionnalités qui utilisent l'API
- Le modèle spaCy français `fr_core_news_md` pour les scripts qui l'utilisent

## Installation

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY

python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

Pour les prototypes basés sur Hugging Face / PyTorch (`code2.py` et `code3.py`), installer aussi :

```bash
pip install -r requirements-advanced.txt
```

## Configuration

1. Copier `.env.example` vers `.env`.
2. Ajouter votre propre clé API dans le fichier `.env`.
3. Configurer les chemins vers Tesseract et Poppler selon votre système.

Exemple de variable d'environnement :

```env
OPENAI_API_KEY=your_api_key_here
```

Ne jamais publier le fichier `.env` ni une clé API sur GitHub.

## Exécution

Les points d'entrée dépendent de la version que vous souhaitez tester. Exemples :

```bash
python code.py
python extraction.py
```

Les scripts `code2.py`, `code3.py` et `cahier.py` sont des prototypes autonomes et peuvent nécessiter une adaptation de leurs chemins locaux et de leur configuration avant exécution.

## Limites connues

- Plusieurs scripts contiennent des chemins Windows locaux à adapter.
- Les dépendances OCR nécessitent des logiciels système (Tesseract et Poppler), pas seulement des paquets Python.
- Les prototypes utilisent des approches et des modèles différents ; les résultats doivent être vérifiés.
- Les documents PDF de test et les données personnelles ne sont pas inclus dans ce dépôt.

## Licence

Aucune licence n'est définie pour le moment. Ajoutez une licence si vous souhaitez autoriser explicitement la réutilisation du code.
