from flask import Flask, request, jsonify
import os
import pytesseract 
from pdf2image import convert_from_path
import fitz
import uuid

app = Flask(__name__)

def is_pdf_text_based(pdf_path, threshold=50, pages_to_check=3):
    """Vérifie si le PDF contient suffisamment de texte extractible"""
    try:
        with fitz.open(pdf_path) as doc:
            total_text = ""
            
            # Vérifie jusqu'à N pages ou jusqu'à atteindre le seuil
            for i in range(min(pages_to_check, doc.page_count)):
                page = doc[i]
                page_text = page.get_text().strip()
                
                # Compte les caractères alphanumériques seulement
                alpha_chars = sum(c.isalnum() for c in page_text)
                total_text += page_text
                
                # Si on atteint le seuil sur une page, on s'arrête
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

@app.route('/api/extract', methods=['POST'])
def extract_pdf():
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
                        "methode": "none",
                        "texte": "",
                        "warning": "Le PDF ne contient aucune page"
                    })
        except Exception as e:
            os.remove(temp_path)
            return jsonify({"error": f"Fichier PDF invalide: {str(e)}"}), 400
        
        # Détection du type de contenu
        is_text_based, char_count = is_pdf_text_based(temp_path)
        
        if is_text_based:
            try:
                texte = extraire_texte_direct(temp_path)
                methode = "direct"
            except Exception as e:
                # Fallback à l'OCR si l'extraction directe échoue
                texte = extraire_texte_ocr(temp_path)
                methode = "ocr (fallback)"
        else:
            texte = extraire_texte_ocr(temp_path)
            methode = "ocr"
        
        os.remove(temp_path)
        return jsonify({
            "methode": methode,
            "texte": texte,
            "detection_info": {
                "caracteres_detectes": char_count,
                "seuil": 50,
                "pages_verifiees": min(3, page_count),
                "est_texte": is_text_based
            }
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500

