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

# Configuration initiale
load_dotenv()

try:
    client = OpenAI()
except:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration des chemins - à adapter selon votre système
POPPLER_PATH = r"C:\Users\ASUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# Chargement du modèle spaCy
try:
    nlp = spacy.load("fr_core_news_md")
    matcher = Matcher(nlp.vocab)
    print("Pipeline spaCy chargé avec succès ✅")
except OSError:
    print("⚠️ Modèle spaCy français non trouvé. Installez-le avec: python -m spacy download fr_core_news_md")
    nlp = None
    matcher = None

print("Pipeline prêt à l'emploi ✅")

# Constantes
CATEGORIES = [
    "exigence administrative",
    "document obligatoire",
    "délai contractuel",
    "condition juridique",
    "attestation requise"
]

KEYWORDS = {
    "administratif": ["joindre", "fournir", "obligatoire", "requis", "attestation", "certificat"],
    "juridique": ["conformité", "légal", "réglementaire", "loi", "article"],
    "délai": ["délai", "échéance", "date limite", "jours ouvrables"]     
}


def parse_json_safe(text):
    """
    Essaie de parser du JSON (provenant de l'IA) même avec ```json ...```.
    Retourne {} si invalide.
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        print("[ERREUR] Type de données inattendu :", type(text))
        return {}

    # Nettoyer les balises markdown
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    cleaned = re.sub(r"^```|```$", "", cleaned.strip(), flags=re.MULTILINE)

    # Extraire le JSON si il y a du texte avant/après
    if not cleaned.strip().startswith("{"):
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            cleaned = m.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[ERREUR] JSON invalide :", e)
        print("Texte reçu après nettoyage :", cleaned[:500])
        return {}


def extraire_texte_ocr(pdf_path):
    """Utilise l'OCR pour extraire le texte d'un PDF scanné"""
    try:
        # Vérifier si Poppler est disponible
        if os.path.exists(POPPLER_PATH):
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        else:
            print("⚠️ Poppler non trouvé au chemin spécifié, tentative sans chemin explicite")
            images = convert_from_path(pdf_path)
        
        texte_total = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='fra')
            texte_total += f"\n\n--- Page {i+1} ---\n\n" + text
        return texte_total
    except Exception as e:
        raise Exception(f"Erreur OCR: {str(e)}")


def extraire_texte_direct(pdf_path):
    """Extrait le texte brut directement d'un PDF numérique"""
    try:
        texte = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texte += page.get_text() + "\n"
        return texte
    except Exception as e:
        raise Exception(f"Erreur extraction directe: {str(e)}")


def preprocess_text(text):
    """Nettoie et segmente le texte en sections"""
    # Supprimer les numéros de page isolés
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Diviser par sections/articles
    sections = re.split(r'\n\s*(ARTICLE|SECTION|CHAPITRE|PARTIE) [IVXLCDM0-9]+\s*[\-\–]?\s*', text)
    return [sec.strip() for sec in sections if sec.strip()]


def detecter_type_pdf(fichier_pdf):
    """Détecte si le PDF est un formulaire ou une image"""
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
    """Structure les clauses par type"""
    data_par_type = defaultdict(list)
    
    for type_clause, textes in clauses.items():
        # Si textes est un dict, on prend uniquement le texte
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
    """
    champs_detectes = []

    # 1. Détection des champs de formulaire (AcroForm)
    try:
        reader = PdfReader(pdf_path)
        fields = reader.get_fields() or {}
        if fields:
            for field_name, field in fields.items():
                champs_detectes.append({
                    "nom": field_name,
                    "type": "formulaire",
                    "page": None  
                })
    except Exception as e:
        print("[ERREUR] Détection champs formulaire :", e)

    # 2. Détection OCR si pas de champs formulaire ou pour compléter
    if not champs_detectes:  # Seulement si pas de champs formulaire
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                if page_num is not None and i != page_num:
                    continue

                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                texte = pytesseract.image_to_string(img, lang="fra")
                lignes = texte.split("\n")
                
                for ligne in lignes:
                    ligne = ligne.strip()
                    if ligne and re.search(r":|_{2,}|…", ligne):
                        champs_detectes.append({
                            "nom": ligne,
                            "type": "texte",
                            "page": i
                        })
            doc.close()
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
    """
    try:
        texte_direct = extraire_texte_direct(pdf_path)
        if len(texte_direct.strip()) >= seuil_detection:
            print("[INFO] PDF numérique détecté, extraction directe")
            return texte_direct
        else:
            print("[INFO] PDF scanné détecté, utilisation de l'OCR")
            return extraire_texte_ocr(pdf_path)
    except Exception as e:
        print(f"[INFO] Extraction directe échouée ({e}), tentative OCR")
        return extraire_texte_ocr(pdf_path)


def extraire_champs_ia(texte_pdf, max_chars=2000):
    """
    Utilise l'IA pour extraire les champs administratifs d'un cahier des charges.
    Retourne un dictionnaire JSON.
    Découpe le texte si trop long.
    """
    if not client:
        print("[ERREUR] Client OpenAI non initialisé")
        return {}

    # Découper le texte si trop long
    chunks = [texte_pdf[i:i+max_chars] for i in range(0, len(texte_pdf), max_chars)]
    champs_admin = {}

    exemple_json = {
        "exigence administrative": "Joindre une attestation de régularité fiscale.",
        "document obligatoire": "Fournir une copie certifiée conforme du registre de commerce.",
        "délai contractuel": "Livraison attendue dans un délai de 30 jours ouvrables.",
        "condition juridique": "Respecter les articles 12 et 15 du code des marchés publics.",
        "attestation requise": "Attestation d'assurance responsabilité civile obligatoire."
    }

    for i, chunk in enumerate(chunks):
        prompt = f"""
        Lis attentivement le texte suivant d'un cahier des charges (section {i+1}/{len(chunks)}). 
        Identifie toutes les clauses et champs administratifs importants 
        (obligations légales, garanties, délais, documents requis, références légales, etc.).

        Retourne STRICTEMENT un objet JSON valide où :
        - chaque clé est un type de clause détectée
        - chaque valeur est le texte exact correspondant
        - si aucune information n'est trouvée pour un type, laisse la valeur vide ""

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
            json_chunk = parse_json_safe(texte_reponse)

            # Fusionner les résultats du chunk
            for k, v in json_chunk.items():
                if v:  # Seulement si la valeur n'est pas vide
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
            champs_admin[k] = " | ".join([str(x) for x in v if x])

    return champs_admin


def mapper_entites_aux_champs(entites, champs):
    """
    Associe chaque entité clé à un champ PDF en fonction de la similarité.
    """
    mapping = {}
    noms_champs = [c["nom"] for c in champs]
    
    for ent in entites:
        if "type" in ent and "texte" in ent:
            meilleur_match = process.extractOne(ent["type"], noms_champs, scorer=fuzz.token_sort_ratio)
            if meilleur_match and meilleur_match[1] > 60:
                mapping[str(meilleur_match[0])] = str(ent["texte"])
    
    return mapping


def valider_champs(mapping, champs_obligatoires):
    """Valide que tous les champs obligatoires sont remplis"""
    erreurs = []
    
    for champ in champs_obligatoires:
        if champ not in mapping or not mapping[champ]:
            erreurs.append(f"Champ obligatoire manquant : {champ}")
    
    return erreurs


def remplir_annexes_intelligent(champs_admin, texte_annexe, champs_detectes=None):
    """
    Remplit les annexes en utilisant l'IA de manière robuste.
    """
    if not champs_admin:
        print("[AVERTISSEMENT] Aucun champ administratif disponible pour remplir l'annexe.")
        return {}

    if not client:
        print("[ERREUR] Client OpenAI non disponible")
        return {}

    prompt = f"""
    Tu es un assistant qui aide à remplir automatiquement une annexe d'un appel d'offres.
    Voici les clauses administratives détectées dans le cahier des charges (JSON) :
    {json.dumps(champs_admin, indent=2, ensure_ascii=False)}
    """

    if champs_detectes:
        champs_info = [{"nom": c["nom"], "type": c["type"]} for c in champs_detectes]
        prompt += f"""

    Voici les champs détectés automatiquement dans l'annexe :
    {json.dumps(champs_info, indent=2, ensure_ascii=False)}
    """

    prompt += f"""

    Voici le texte brut de l'annexe à remplir :
    \"\"\"{texte_annexe[:1500]}\"\"\"

    Ta tâche :
    - Identifie les champs à remplir dans ce texte d'annexe.
    - Remplis-les avec les informations les plus pertinentes depuis les clauses administratives
      et/ou les champs détectés automatiquement (si présents).
    - Si une information n'est pas trouvée, laisse la valeur vide "".
    - Réponds STRICTEMENT en JSON valide : {{"nom_champ": "valeur"}}.
    - Ne retourne rien d'autre (pas de texte explicatif).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un assistant spécialisé en appels d'offres. Tu retournes uniquement du JSON valide."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        texte_reponse = response.choices[0].message.content.strip()
        mapping = parse_json_safe(texte_reponse)
        
        if isinstance(mapping, dict):
            print("[INFO] Annexe remplie avec succès ✅")
            return mapping
        else:
            print("[ERREUR] Réponse IA non conforme (pas un dict).")
            return {}

    except Exception as e:
        print(f"[ERREUR] Appel API OpenAI échoué : {e}")
        return {}


def verifier_coherence(mapping):
    """Vérifie la cohérence des données remplies"""
    incoherences = []

    # Validation des formats de date
    date_regex = r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b"
    for cle, valeur in mapping.items():
        if "date" in cle.lower() and valeur and not re.search(date_regex, str(valeur)):
            incoherences.append(f"Format de date invalide pour le champ '{cle}': {valeur}")

    # Validation CIN tunisien (8 chiffres)
    for cle, valeur in mapping.items():
        if "cin" in cle.lower() and valeur and not re.match(r"^\d{8}$", str(valeur)):
            incoherences.append(f"Numéro CIN invalide pour '{cle}' (doit contenir 8 chiffres): {valeur}")

    return incoherences


def sauvegarder_annexe_intelligent(chemin_pdf, mapping_final, sortie_pdf):
    """
    Remplit directement le PDF original avec les valeurs de mapping_final.
    """
    try:
        doc = fitz.open(chemin_pdf)

        for num_page in range(len(doc)):
            page = doc[num_page]
            
            # Méthode 1: Essayer de remplir les champs de formulaire
            try:
                widgets = page.widgets()
                for widget in widgets:
                    field_name = widget.field_name
                    if field_name in mapping_final:
                        widget.field_value = mapping_final[field_name]
                        widget.update()
            except Exception as e:
                print(f"[INFO] Pas de champs formulaire sur page {num_page}: {e}")

            # Méthode 2: Recherche et remplacement de texte
            blocks = page.get_text("blocks")
            for b in blocks:
                if len(b) >= 4:
                    x0, y0, x1, y1 = b[:4]
                    texte_block = b[4] if len(b) > 4 else ""
                    
                    # Vérifier si ce bloc correspond à une clé dans le mapping
                    for cle, valeur in mapping_final.items():
                        if cle.lower() in texte_block.lower() and valeur:
                            rect = fitz.Rect(x0, y0, x1, y1)
                            # Effacer le texte existant
                            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                            # Écrire la nouvelle valeur
                            page.insert_text(
                                fitz.Point(x0, y0 + 12),
                                str(valeur),
                                fontsize=10,
                                color=(0, 0, 0)
                            )

        doc.save(sortie_pdf)
        doc.close()
        print(f"[INFO] Annexe sauvegardée : {sortie_pdf}")

    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder l'annexe : {e}")
        raise


def traiter_pdf(file):
    """Fonction principale pour traiter le PDF via Gradio"""
    if not file:
        return None, "⚠️ Veuillez uploader un PDF."

    try:
        # 1. Extraire texte
        print(f"[INFO] Traitement du fichier: {file.name}")
        texte_pdf = extraire_texte_pdf(file.name)
        
        if not texte_pdf.strip():
            return None, "⚠️ Impossible d'extraire du texte du PDF."

        print(f"[INFO] Texte extrait: {len(texte_pdf)} caractères")

        # 2. Champs admin via IA
        champs_admin = extraire_champs_ia(texte_pdf)
        print(f"[INFO] Champs administratifs détectés: {len(champs_admin)}")

        # 3. Structuration
        clauses_structurees = structurer_clauses(champs_admin)

        # 4. Détection des zones
        champs_detectes = detecter_champs_pdf(file.name)
        print(f"[INFO] Champs à remplir détectés: {len(champs_detectes)}")

        # 5. Mapping IA
        mapping_final = remplir_annexes_intelligent(champs_admin, texte_pdf[:1500], champs_detectes)
        print(f"[INFO] Mapping final: {len(mapping_final)} champs")

        # 6. Validation
        champs_obligatoires = []  # À définir selon vos besoins
        erreurs = valider_champs(mapping_final, champs_obligatoires)
        
        # Vérification de cohérence
        incoherences = verifier_coherence(mapping_final)
        if incoherences:
            erreurs.extend(incoherences)

        # 7. Sauvegarde
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as sortie_temp:
            sortie_pdf = sortie_temp.name

        sauvegarder_annexe_intelligent(file.name, mapping_final, sortie_pdf)

        # Résumé des erreurs si manquants
        if erreurs:
            message = "⚠️ Problèmes détectés : " + "; ".join(erreurs)
        elif not mapping_final:
            message = "⚠️ Aucun champ rempli automatiquement."
        else:
            message = f"✅ Annexe remplie avec succès ! {len(mapping_final)} champs traités."

        return sortie_pdf, message

    except Exception as e:
        print(f"[ERREUR] Traitement PDF échoué: {e}")
        return None, f"❌ Erreur : {str(e)}"


def main_standalone():
    """Fonction pour test en mode standalone - MODIFIEZ LE CHEMIN CI-DESSOUS"""
    # ⚠️ CHANGEZ CE CHEMIN VERS VOTRE FICHIER PDF DE TEST
    chemin_pdf = input("Entrez le chemin complet de votre PDF de test: ").strip()
    
    if not chemin_pdf:
        print("[ERREUR] Aucun chemin fourni.")
        return
        
    if not os.path.exists(chemin_pdf):
        print(f"[ERREUR] Fichier non trouvé : {chemin_pdf}")
        return

    # Générer le nom de sortie automatiquement
    base_name = os.path.splitext(chemin_pdf)[0]
    sortie_pdf = f"{base_name}_rempli.pdf"
    
    print(f"[INFO] Traitement de : {chemin_pdf}")
    print(f"[INFO] Sortie vers : {sortie_pdf}")

    try:
        # 1. Extraire le texte complet du PDF
        texte_pdf = extraire_texte_pdf(chemin_pdf)
        if not texte_pdf.strip():
            print("[ERREUR] Aucun texte extrait du PDF.")
            return

        print(f"[INFO] Texte extrait: {len(texte_pdf)} caractères")

        # 2. Extraire les champs administratifs via l'IA
        champs_admin = extraire_champs_ia(texte_pdf)
        if not isinstance(champs_admin, dict):
            champs_admin = {}

        print("\n=== Champs administratifs détectés ===")
        print(json.dumps(champs_admin, indent=4, ensure_ascii=False))

        # 3. Structuration des clauses par type
        clauses_structurees = structurer_clauses(champs_admin)
        print("\n=== Clauses structurées par type ===")
        for type_clause, textes in clauses_structurees.items():
            print(f"- {type_clause.upper()} :")
            for t in textes:
                print(f"   • {t}")

        # 4. Détecter les champs à remplir
        champs_detectes = detecter_champs_pdf(chemin_pdf)
        print(f"\n=== Zones à remplir détectées ===")
        print(f"{len(champs_detectes)} champs trouvés")
        for champ in champs_detectes[:5]:  # Afficher les 5 premiers
            print(f"- {champ}")

        # 5. Mapping
        # 5a. Mapping automatique classique
        entites = [{"type": k, "texte": v} for k, v in champs_admin.items()]
        mapping_auto = mapper_entites_aux_champs(entites, champs_detectes)
        print(f"\n[INFO] Mapping automatique: {len(mapping_auto)} correspondances")

        # 5b. Mapping avec IA
        mapping_ia = remplir_annexes_intelligent(champs_admin, texte_pdf, champs_detectes)
        print(f"[INFO] Mapping IA: {len(mapping_ia)} correspondances")

        # Fusionner les deux mappages
        mapping_final = {**mapping_auto, **mapping_ia}
        print(f"\n=== Mapping final ===")
        print(json.dumps(mapping_final, indent=2, ensure_ascii=False))

        # 6. Validation des champs obligatoires
        champs_obligatoires = []  # À compléter selon vos besoins
        erreurs = valider_champs(mapping_final, champs_obligatoires)
        incoherences = verifier_coherence(mapping_final)
        
        if erreurs:
            print("\n[⚠️] Erreurs de validation :", erreurs)
        if incoherences:
            print("\n[⚠️] Incohérences détectées :", incoherences)

        # 7. Sauvegarder le PDF rempli
        if mapping_final:
            try:
                sauvegarder_annexe_intelligent(chemin_pdf, mapping_final, sortie_pdf)
                print(f"\n[SUCCÈS] PDF rempli généré : {sortie_pdf}")
            except Exception as e:
                print(f"[ERREUR] Impossible de sauvegarder le PDF final : {e}")
        else:
            print("[AVERTISSEMENT] Aucun champ à remplir trouvé.")

    except Exception as e:
        print(f"[ERREUR] Traitement échoué : {e}")


# Interface Gradio
def create_gradio_interface():
    """Crée l'interface Gradio"""
    demo = gr.Interface(
        fn=traiter_pdf,
        inputs=gr.File(
            type="filepath", 
            file_types=[".pdf"], 
            label="Uploader votre Cahier des Charges"
        ),
        outputs=[
            gr.File(label="Télécharger l'annexe remplie"),
            gr.Textbox(label="Statut / Alertes", lines=3)
        ],
        title="🤖 Assistant Cahier des Charges → Annexe remplie",
        description="Uploadez un PDF de cahier des charges. Le système extrait les champs administratifs, détecte les zones à remplir et génère une annexe remplie automatiquement.",
        examples=[],
        allow_flagging="never"
    )
    return demo


if __name__ == "__main__":
    import sys
    
    # Vérifier les dépendances critiques
    missing_deps = []
    try:
        import fitz
    except ImportError:
        missing_deps.append("PyMuPDF")
    
    try:
        import pytesseract
    except ImportError:
        missing_deps.append("pytesseract")
    
    if missing_deps:
        print(f"❌ Dépendances manquantes: {', '.join(missing_deps)}")
        print("Installez-les avec: pip install PyMuPDF pytesseract")
        sys.exit(1)

    # Choisir le mode d'exécution
    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        print("🔧 Mode standalone")
        main_standalone()
    else:
        print("🌐 Lancement de l'interface Gradio")
        print("📁 Vous pourrez uploader votre PDF directement dans l'interface web")
        demo = create_gradio_interface()
        
        # Configuration réseau corrigée
        try:
            demo.launch(
                server_name="127.0.0.1",  # localhost au lieu de 0.0.0.0
                server_port=7860,
                share=False,
                debug=False,  # Désactiver le debug pour éviter les conflits
                show_error=True,
                quiet=False
            )
        except Exception as e:
            print(f"❌ Erreur de lancement Gradio: {e}")
            print("🔄 Tentative sur un port différent...")
            try:
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=7861,  # Port alternatif
                    share=False,
                    debug=False
                )
            except Exception as e2:
                print(f"❌ Impossible de lancer l'interface: {e2}")
                print("💡 Essayez de redémarrer votre terminal ou utiliser le mode standalone avec:")
                print("python votre_script.py --standalone")