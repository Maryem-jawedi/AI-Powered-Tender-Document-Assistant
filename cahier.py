import fitz
import os
import pytesseract
from pdf2image import convert_from_path
import re 
from collections import defaultdict
import spacy
from spacy.matcher import Matcher
import json
import io
import openai
from openai import OpenAI
from PIL import ImageDraw, ImageFont, Image
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from PyPDF2 import PdfReader
import gradio as gr
import tempfile

load_dotenv()
try:
    client = OpenAI()
except:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
POPPLER_PATH = r"C:\Users\ASUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
nlp= spacy.load("fr_core_news_md") 
matcher = Matcher(nlp.vocab)    
print("Pipeline prêt à l’emploi ✅")

CATEGORIES = [
    "exigence administrative",
    "document obligatoire",
    "délai contractuel",
    "condition juridique",
    "attestation requise"
]

KEYWORDS = {
    "administratif": ["joindre", "fournir" , 
                      "obligatoire", "requis" , "attestation", "certificat"],
    "juridique" : ["conformité", "légal" , "réglementaire", "loi", "article"],
    "délai": ["délai", "échéance", "date limite", "jours ouvrables"]     
} 

def parse_json_safe(text):
    """
    Essaie de parser du JSON (provenant de l’IA) même avec ```json ...```.
    Retourne {} si invalide.
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        print("[ERREUR] Type de données inattendu :", type(text))
        return {}

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    cleaned = re.sub(r"^```|```$", "", cleaned.strip(), flags=re.MULTILINE)

    if not cleaned.strip().startswith("{"):
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            cleaned = m.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[ERREUR] JSON invalide :", e)
        print("Texte reçu après nettoyage :", cleaned[:1000])
        return {}


def extraire_texte_ocr(pdf_path):
    """Utilise l’OCR pour extraire le texte d’un PDF scanné"""
    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:\Users\ASUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")
        texte_total = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='fra')
            texte_total += f"\n\n--- Page {i+1} ---\n\n" + text
        return texte_total
    except Exception as e:
        raise Exception(f"Erreur OCR: {str(e)}")


def extraire_texte_direct(pdf_path):
    """Extrait le texte brut directement d’un PDF numérique"""
    try:
        texte = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texte += page.get_text() + "\n"
        return texte
    except Exception as e:
        raise Exception(f"Erreur extraction directe: {str(e)}")
    


def preprocess_text(text):
    """netoie et segmente le texte en sections"""
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    sections= re.split(r'\n\s*(ARTICLE|SECTION|CHAPITRE|PARTIE) [IVXLCDM0-9]+\s*[\-\–]?\s*', text)
    return [sec.strip() for sec in sections if sec.strip()]

           


def detecter_type_pdf(fichier_pdf):
    from PyPDF2 import PdfReader
    try:
        reader = PdfReader(fichier_pdf)
        champs = reader.get_fields()
        if champs:
            return "formulaire"
        else:
            return "image"
    except Exception as e:
        print("Erreur lors de la détection du type de PDF :", e)
        return "inconnu"    

def structurer_clauses(clauses):
    data_par_type = defaultdict(list)
    for type_clause, textes in clauses.items():
        # si textes est un dict, on prend uniquement le texte
        if isinstance(textes, dict):
            textes = [textes.get("texte", "")]
        elif isinstance(textes, list):
            textes = [t if isinstance(t, str) else t.get("texte", "") for t in textes]
        else:
            textes = [str(textes)]
        
        for texte in textes:
            texte = texte.strip()
            if texte:
                data_par_type[type_clause.lower().strip()].append(texte)
    return data_par_type


def detecter_champs_pdf(pdf_path, page_num=None):
    """
    Détecte les champs ou zones à remplir dans un PDF.
    1. Détection des champs de formulaire (AcroForm) si présents.
    2. Sinon, détection par OCR des zones à remplir (texte avec ':' ou '____').
    :param pdf_path: chemin du PDF
    :param page_num: numéro de la page à analyser (None pour toutes)
    :return: liste de dictionnaires {nom, type, page}
    """
    champs_detectes = []

    # 1. Détection des champs de formulaire (AcroForm)
    try:
        reader = PdfReader(pdf_path)
        fields = reader.get_fields() or {}
        if fields:
            for field in fields.values():
                champs_detectes.append({
                    "nom": field.get('/T', 'Champ sans nom'),
                    "type": "formulaire",
                    "page": None  
                })
    except Exception as e:
        print("[ERREUR] Détection champs formulaire :", e)

    # 2. Détection OCR si pas de champs formulaire ou pour compléter
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            if page_num is not None and i != page_num:
                continue  # ignorer les pages non demandées

            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            texte = pytesseract.image_to_string(img, lang="fra")
            lignes = texte.split("\n")
            for ligne in lignes:
                if re.search(r":|_{2,}|…", ligne):
                    champs_detectes.append({
                        "nom": ligne.strip(),
                        "type": "texte",
                        "page": i
                    })
    except Exception as e:
        print("[ERREUR] OCR sur PDF :", e)

    # Filtrer par page si page_num est spécifié
    if page_num is not None:
        champs_page = [c for c in champs_detectes if c.get("page") == page_num or c.get("page") is None]
        return champs_page

    return champs_detectes

   
def extraire_texte_pdf(pdf_path, seuil_detection=50):
    """
    Détecte si un PDF est scanné ou texte et extrait le texte.
    :param pdf_path: chemin du fichier PDF
    :param seuil_detection: nombre minimum de caractères pour considérer un PDF comme texte
    :return: texte extrait
    """
    try:
        texte_direct = extraire_texte_direct(pdf_path)
        if len(texte_direct.strip()) >= seuil_detection:
            return texte_direct
        else:
            return extraire_texte_ocr(pdf_path)
    except Exception as e:
        return extraire_texte_ocr(pdf_path)

def extraire_champs_ia(texte_pdf, max_chars=2000):
    """
    Utilise l’IA pour extraire les champs administratifs d’un cahier des charges.
    Retourne un dictionnaire JSON.
    Découpe le texte si trop long.
    """
    # Découper le texte si trop long
    chunks = [texte_pdf[i:i+max_chars] for i in range(0, len(texte_pdf), max_chars)]
    champs_admin = {}

    exemple_json = {
        "exigence administrative": "Joindre une attestation de régularité fiscale.",
        "document obligatoire": "Fournir une copie certifiée conforme du registre de commerce.",
        "délai contractuel": "Livraison attendue dans un délai de 30 jours ouvrables.",
        "condition juridique": "Respecter les articles 12 et 15 du code des marchés publics.",
        "attestation requise": "Attestation d’assurance responsabilité civile obligatoire."
    }

    for i, chunk in enumerate(chunks):
        prompt = f"""
        Lis attentivement le texte suivant d’un cahier des charges (section {i+1}/{len(chunks)}). 
        Identifie toutes les clauses et champs administratifs importants 
        (obligations légales, garanties, délais, documents requis, références légales, etc.).

        Retourne STRICTEMENT un objet JSON valide où :
        - chaque clé est un type de clause détectée
        - chaque valeur est le texte exact correspondant
        - si aucune information n’est trouvée pour un type, laisse la valeur vide ""

        Exemple de format JSON attendu :
        {json.dumps(exemple_json, indent=2, ensure_ascii=False)}

        Texte à analyser :
        \"\"\"{chunk}\"\"\"
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": "Tu es un assistant qui retourne uniquement du JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            texte_reponse = response.choices[0].message.content
            # Extraire le JSON même si encodé avec ```json
            json_chunk = parse_json_safe(texte_reponse)

            # Fusionner les résultats du chunk
            for k, v in json_chunk.items():
                if k in champs_admin and champs_admin[k]:
                    if isinstance(champs_admin[k], list):
                        champs_admin[k].append(v)
                    else:
                        champs_admin[k] = [champs_admin[k], v]
                else:
                    champs_admin[k] = v

        except Exception as e:
            print(f"[ERREUR] Extraction IA section {i+1} échouée :", e)

    # Nettoyer les valeurs : convertir les listes en texte concaténé
    for k, v in champs_admin.items():
        if isinstance(v, list):
            champs_admin[k] = " ".join([str(x) for x in v if x])

    return champs_admin


def mapper_entites_aux_champs(entites, champs):
    """
    Associe chaque entité clé à un champ PDF en fonction de la similarité.
    """
    mapping = {}
    noms_champs = [c["nom"] for c in champs]
    for ent in entites:
        meilleur_match = process.extractOne(ent["type"], noms_champs, scorer=fuzz.token_sort_ratio)
        if meilleur_match and meilleur_match[1] > 60:
            mapping[str(meilleur_match[0])] = str(ent["texte"])
    return mapping



def valider_champs(mapping, champs_obligatoires):
    erreurs = []

    for champ in champs_obligatoires:
        if champ not in mapping or not mapping[champ]:
            erreurs.append(f"Champ obligatoire manquant : {champ}")

    return erreurs
def remplir_annexes_intelligent(champs_admin, texte_annexe, champs_detectes=None):
    """
    Remplit les annexes en utilisant l'IA de manière robuste.
    - champs_admin : dictionnaire des clauses administratives extraites
    - texte_annexe : texte brut de l'annexe à remplir
    - champs_detectes : dictionnaire optionnel des champs détectés automatiquement dans l'annexe
    Retourne : un mapping champ → valeur (dict)
    """
    if not champs_admin:
        print("[AVERTISSEMENT] Aucun champ administratif disponible pour remplir l'annexe.")
        return {}

    prompt = f"""
    Tu es un assistant qui aide à remplir automatiquement une annexe d'un appel d'offres.
    Voici les clauses administratives détectées dans le cahier des charges (JSON) :
    {json.dumps(champs_admin, indent=2, ensure_ascii=False)}
    """

    if champs_detectes:
        prompt += f"""

    Voici les champs détectés automatiquement dans l'annexe (OCR / extraction de formulaire) :
    {json.dumps(champs_detectes, indent=2, ensure_ascii=False)}
    """

    prompt += f"""

    Voici le texte brut de l'annexe à remplir :
    \"\"\"{texte_annexe}\"\"\"

    Ta tâche :
    - Identifie les champs à remplir dans ce texte d'annexe.
    - Remplis-les avec les informations les plus pertinentes depuis les clauses administratives
      et/ou les champs détectés automatiquement (si présents).
    - Si une information n’est pas trouvée, laisse la valeur vide "".
    - Réponds STRICTEMENT en JSON valide : {{"nom_champ": "valeur"}}.
    - Ne retourne rien d'autre (pas de texte explicatif).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un assistant spécialisé en appels d'offres."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        texte_reponse = response.choices[0].message.content.strip()

        try:
            mapping = json.loads(texte_reponse)
            if isinstance(mapping, dict):
                print("[INFO] Annexe remplie avec succès ✅")
                return mapping
            else:
                print("[ERREUR] Réponse IA non conforme (pas un dict).")
                return {}

        except json.JSONDecodeError:
            print("[ERREUR] Impossible de parser le JSON renvoyé par l'IA :")
            print(texte_reponse)
            return {}

    except Exception as e:
        print(f"[ERREUR] Appel API OpenAI échoué : {e}")
        return {}



def verifier_coherence(mapping):
    incoherences = []

    date_regex = r"\b\d{2}/\d{2}/\d{4}\b"
    if "Date" in mapping and not re.match(date_regex, str(mapping["Date"])):
        incoherences.append("Format de date invalide pour le champ 'Date'")

    if "CIN" in mapping and not re.match(r"^\d{8}$", str(mapping["CIN"])):
        incoherences.append("Numéro CIN invalide (doit contenir 8 chiffres)")

    return incoherences

def sauvegarder_annexe_intelligent(chemin_pdf, mapping_final, sortie_pdf):
    """
    Remplit directement le PDF original avec les valeurs de mapping_final.
    Chaque champ détecté est remplacé par la valeur correspondante.
    """
    try:
        doc = fitz.open(chemin_pdf)

        for num_page in range(len(doc)):
            page = doc[num_page]
            blocks = page.get_text("blocks")

            for b in blocks:
                x0, y0, x1, y1, text, _, _, _ = b
                texte_block = text.strip()

                # Vérifier si ce bloc correspond à une clé dans le mapping
                for cle, valeur in mapping_final.items():
                    if cle.lower() in texte_block.lower():
                        rect = fitz.Rect(x0, y0, x1, y1)
                        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                        
                        # Écrire la nouvelle valeur
                        page.insert_text(
                            fitz.Point(x0, y0 + 10),
                            valeur,
                            fontsize=10,
                            color=(0, 0, 0)
                        )

        doc.save(sortie_pdf)
        doc.close()

        print(f"[INFO] Annexe sauvegardée : {sortie_pdf}")

    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder l’annexe : {e}")

def traiter_pdf(file):
    if not file:
        return None, "⚠️ Veuillez uploader un PDF."

    try:
        # 1. Extraire texte
        texte_pdf = extraire_texte_pdf(file.name)
        if not texte_pdf.strip():
            return None, "⚠️ Impossible d'extraire du texte du PDF."

        # 2. Champs admin via IA
        champs_admin = extraire_champs_ia(texte_pdf)

        # 3. Structuration
        clauses_structurees = structurer_clauses(champs_admin)

        # 4. Détection des zones
        champs_detectes = detecter_champs_pdf(file.name, page_num=None)

        # 5. Mapping IA
        mapping_final = remplir_annexes_intelligent(champs_admin, texte_pdf, champs_detectes)

        # 6. Validation
        champs_obligatoires = []  # tu peux définir une liste ici
        erreurs = valider_champs(mapping_final, champs_obligatoires)

        # 7. Sauvegarde
        sortie_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        sortie_pdf = sortie_temp.name
        sauvegarder_annexe_intelligent(file.name, mapping_final, sortie_pdf)
        # Résumé des erreurs si manquants
        if erreurs:
            return sortie_pdf, "⚠️ Champs obligatoires manquants : " + "; ".join(erreurs)
        elif not mapping_final:
            return sortie_pdf, "⚠️ Aucun champ rempli automatiquement."
        else:
            return sortie_pdf, "✅ Annexe remplie avec succès !"

    except Exception as e:
        return None, f"[ERREUR] {str(e)}"

# Interface Gradio
demo = gr.Interface(
    fn=traiter_pdf,
    inputs=gr.File(type="filepath", file_types=[".pdf"], label="Uploader votre Cahier des Charges"),
    outputs=[
        gr.File(label="Télécharger l'annexe remplie"),
        gr.Textbox(label="Statut / Alertes")
    ],
    title="Assistant Cahier des Charges → Annexe remplie",
    description="Uploadez un PDF de cahier des charges. Le système extrait les champs administratifs, détecte les zones à remplir et génère une annexe remplie automatiquement."
)





if __name__ == "__main__":

    demo.launch()
    chemin_pdf = "C:/Users/ASUS/Desktop/projet ia/testCC.pdf"
    sortie_pdf = "C:/Users/ASUS/Desktop/projet ia/testCC_rempli.pdf"


    # 1. Extraire le texte complet du PDF
    texte_pdf = extraire_texte_pdf(chemin_pdf)
    if not texte_pdf.strip():
        print("[ERREUR] Aucun texte extrait du PDF.")
        exit(1)



    # 2. Extraire les champs administratifs via l’IA
    champs_admin = extraire_champs_ia(texte_pdf)
    if not isinstance(champs_admin, dict) or not champs_admin:
        print("[AVERTISSEMENT] Aucun champ administratif détecté")
        champs_admin = {}

    print("Champs administratifs détectés :")
    print(json.dumps(champs_admin, indent=4, ensure_ascii=False))
    

    # 3. Structuration des clauses par type
    clauses_structurees = structurer_clauses(champs_admin)
    print("\nClauses structurées par type :")
    for type_clause, textes in clauses_structurees.items():
        print(f"- {type_clause.upper()} :")
        for t in textes:
            print(f"   • {t}")


    # 4. detecter les champs à remplir
    champs_detectes = detecter_champs_pdf(chemin_pdf, page_num=None)
    print(f"\nZones à remplir détectées : {len(champs_detectes)} champs trouvés")
   
   
    # 5. mapping
     # 5a. Mapping automatique classique (optionnel)
    entites = [{"type": k, "texte": v} for k, v in champs_admin.items()]
    mapping_auto = mapper_entites_aux_champs(entites, champs_detectes)
    print("[INFO] Mapping automatique basé sur similarité :", mapping_auto)

    # 5b. Mapping avec IA (plus robuste)
    mapping_ia = remplir_annexes_intelligent(champs_admin, texte_pdf, champs_detectes)

    # Fusionner les deux mappages si nécessaire
    mapping_final = {**mapping_auto, **mapping_ia}


    # 6. Validation des champs obligatoires (optionnel)
    champs_obligatoires = []  # à compléter si nécessaire
    erreurs = valider_champs(mapping_final, champs_obligatoires)
    if erreurs:
        print("[⚠️] Erreurs de validation :", erreurs)


    # 7. Sauvegarder le PDF rempli
    if mapping_final:
        try:
            sauvegarder_annexe_intelligent(chemin_pdf, mapping_final, sortie_pdf)
            print(f"[SUCCÈS] PDF rempli généré : {sortie_pdf}")
        except Exception as e:
            print(f"[ERREUR] Impossible de sauvegarder le PDF final : {e}")
    else:
        print("[AVERTISSEMENT] Aucun champ à remplir trouvé.")






    
