from flask import Flask, request, jsonify
import os
import pytesseract 
from pdf2image import convert_from_path
import fitz
import uuid
from openai import OpenAI
import openai
import json
from dotenv import load_dotenv
# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY non configurée dans les variables d'environnement")
# Configuration Tesseract (OCR)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# Modèle OpenAI à utiliser
OPENAI_MODEL = "gpt-3.5-turbo"  

def is_pdf_text_based(pdf_path, threshold=50, pages_to_check=3):
    """Vérifie si le PDF contient suffisamment de texte extractible"""
    try:
        with fitz.open(pdf_path) as doc:
            total_text = ""
            for i in range(min(pages_to_check, doc.page_count)):
                page = doc[i]
                page_text = page.get_text().strip()
                alpha_chars = sum(c.isalnum() for c in page_text)
                total_text += page_text
                if alpha_chars > threshold:
                    return True, len(total_text)
            return False, len(total_text)
    except Exception:
        return False, 0
    
def extraire_texte_direct(pdf_path):
    try:
        texte = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texte += page.get_text() + "\n"
        return texte
    except Exception as e:
        raise Exception(f"Erreur extraction directe: {str(e)}")

def extraire_texte_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        texte_total = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='fra')
            texte_total += f"\n\n--- Page {i+1} ---\n\n" + text
        return texte_total
    except Exception as e:
        raise Exception(f"Erreur OCR: {str(e)}")

def structurer_cv_avec_openai(texte_cv, format_json_attendu=None):
    """
    Transforme le texte brut d'un CV en JSON structuré avec OpenAI.
    - texte_cv : texte brut extrait du CV
    - format_json_attendu : (optionnel) chaîne décrivant le format JSON attendu
    """
    try:
        # Prompt dynamique
        if format_json_attendu:
            prompt = (
                "Tu es un expert en traitement automatique de CV.\n"
                "Analyse le texte ci-dessous extrait d'un CV et structure les informations dans le format JSON strict suivant :\n\n"
                f"{format_json_attendu}\n\n"
                "Respecte le format mot à mot, sans explication. Si une information est absente, mets une chaîne vide \"\" ou une liste vide [].\n\n"
                "Voici le contenu du CV à analyser :\n\n"
                f"{texte_cv[:10000]}"
            )
        else:
            # Prompt par défaut
            prompt = (
                "Vous êtes un expert en extraction d'informations de CV. "
                "Analysez le texte suivant d'un CV et extrayez les informations suivantes "
                "dans un format JSON strict avec ces clés:\n"
                "- nom_complet (string)\n"
                "- email (string)\n"
                "- telephone (string)\n"
                "- competences (liste de strings)\n"
                "- experiences (liste d'objets avec: poste, entreprise, duree, description)\n"
                "- formations (liste d'objets avec: diplome, etablissement, annee)\n"
                "- langues (liste de strings)\n\n"
                "Texte du CV:\n"
                f"{texte_cv[:10000]}"
            )

        # Appel à l'API OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Tu extraits des informations de CV et les structures en JSON avec rigueur."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        json_str = response.choices[0].message.content
        return json.loads(json_str)

    except Exception as e:
        raise Exception(f"Erreur OpenAI: {str(e)}")


@app.route('/api/extract-cv', methods=['POST'])
def extract_and_structure_cv():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier PDF reçu"}), 400
    
    temp_path = f"temp_upload_{uuid.uuid4().hex}.pdf"
    request.files['file'].save(temp_path)

    try:
        # Vérification initiale du PDF
        try:
            with fitz.open(temp_path) as doc:
                page_count = doc.page_count
                if page_count == 0:
                    return jsonify({
                        "error": "Le PDF ne contient aucune page"
                    }), 400
        except Exception as e:
            os.remove(temp_path)
            return jsonify({"error": f"Fichier PDF invalide: {str(e)}"}), 400
        
        # Extraction du texte brut
        is_text_based, char_count = is_pdf_text_based(temp_path)
        
        if is_text_based:
            try:
                texte_brut = extraire_texte_direct(temp_path)
                methode = "direct"
            except Exception as e:
                texte_brut = extraire_texte_ocr(temp_path)
                methode = "ocr (fallback)"
        else:
            texte_brut = extraire_texte_ocr(temp_path)
            methode = "ocr"
        
        # Suppression du fichier temporaire
        os.remove(temp_path)
        
        # Structuration avec OpenAI
        cv_structuré = structurer_cv_avec_openai(texte_brut)
        
        return jsonify({
            "methode_extraction": methode,
            "cv_json": cv_structuré,
            "metadata": {
                "caracteres_extraits": len(texte_brut),
                "pages_verifiees": min(3, page_count)
            }
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Vérification finale de la configuration
    if not OPENAI_API_KEY:
        print("❌ ERREUR: OPENAI_API_KEY non configurée dans .env!")
        exit(1)
    else:
        print("✅ Configuration chargée avec succès")
        print(f"🔑 OpenAI key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")
    
    if TESSERACT_CMD:
        print(f"🔍 Tesseract path: {TESSERACT_CMD}")
    
    # Message de démarrage
    print("\n" + "="*50)
    print("🚀 Application d'extraction de CV démarrée")
    print(f"🌐 Point de terminaison: POST http://localhost:5000/api/extract-cv")
    print("="*50 + "\n")
    
    # Démarrer le serveur
    app.run(host='0.0.0.0', port=5000, debug=True)