"""
ao_llm_pipeline.py

Pipeline:
1) Extraction texte (fitz -> OCR fallback)
2) Détection des champs de l'annexe (AcroForm) ou repérage des pages "annexes"
3) Envoi du contexte au LLM (local ou via API) pour produire un mapping {champ: valeur}
4) Remplissage du PDF: AcroForm (si présent) ou insertion texte sur pages images
"""

import os
import re
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import spacy

# PDF form libraries
from pdfrw import PdfReader as PdfrwReader, PdfWriter as PdfrwWriter, PdfDict
from PyPDF2 import PdfReader as PyPDF2Reader, PdfWriter as PyPDF2Writer

# Image drawing
from PIL import ImageDraw, ImageFont

# LLM backends
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional: openai SDK (if using API mode)
# import openai

# -------------------------
# CONFIG & LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AO-LLM")

POPPLER_PATH = r"C:\Users\ASUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
TESSERACT_LANG = "fra"
TESSERACT_PSM = "--psm 6 --oem 3"

# LLM configuration: choose 'local' or 'api'
LLM_BACKEND = "local"  # options: "local", "openai" (or other)
LOCAL_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # change to LLaMA 3 id if available locally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If using API mode (OpenAI or other), set API key via env var externally
# os.environ["OPENAI_API_KEY"] = "sk-..."

# -------------------------
# UTIL: robust JSON extraction from model output
# -------------------------
def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Récupère le premier objet JSON contenu dans text.
    Robuste : cherche la première '{' et la dernière '}' correspondante.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    # find matching closing brace
    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    # try to clean trailing commas common from LLMs
                    cleaned = re.sub(r",\s*}", "}", candidate)
                    cleaned = re.sub(r",\s*\]", "]", cleaned)
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        return None
    return None

# -------------------------
# EXTRACTION TEXTE (direct + OCR fallback)
# -------------------------
def extraire_texte_direct(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def extraire_texte_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
    all_text = ""
    for i, im in enumerate(images):
        txt = pytesseract.image_to_string(im, lang=TESSERACT_LANG, config=TESSERACT_PSM)
        all_text += f"\n\n--- Page {i+1} ---\n\n" + txt
    return all_text

def extraire_texte_intelligent(pdf_path: str) -> Tuple[str, dict]:
    """
    Essaie extraction directe; si insuffisant (ou si erreurs), OCR.
    Retourne texte et metadata.
    """
    metadata = {"method": "direct"}
    try:
        text = extraire_texte_direct(pdf_path)
        # heuristique : si très court -> OCR
        if len(text.strip()) < 200:
            logger.info("Texte direct trop court -> OCR")
            text = extraire_texte_ocr(pdf_path)
            metadata["method"] = "ocr"
    except Exception as e:
        logger.warning(f"Erreur extraction directe: {e} -> OCR")
        text = extraire_texte_ocr(pdf_path)
        metadata["method"] = "ocr_fallback"
    return text, metadata

# -------------------------
# DÉTECTION CHAMPS (AcroForm) & PAGES ANNEXES
# -------------------------
def detecter_champs_acroform(pdf_path: str) -> List[str]:
    """Retourne la liste des noms de champs si AcroForm présent"""
    try:
        reader = PyPDF2Reader(pdf_path)
        fields = reader.get_fields()
        if not fields:
            return []
        return list(fields.keys())
    except Exception as e:
        logger.warning(f"Impossible lire AcroForm via PyPDF2: {e}")
        return []

def detecter_pages_annexes(pdf_path: str) -> List[int]:
    """
    Recherche occurrences du mot "annexe" (ou variantes) dans le texte / OCR
    Retourne index des pages probables.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, p in enumerate(pdf.pages):
                t = p.extract_text() or ""
                if "annexe" in t.lower():
                    pages.append(i)
    except Exception as e:
        logger.debug(f"pdfplumber non dispo pour detection annexe: {e}")

    # fallback OCR si aucune page trouvée
    if not pages:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=200)
        for i, im in enumerate(images):
            txt = pytesseract.image_to_string(im, lang=TESSERACT_LANG, config=TESSERACT_PSM)
            if "annexe" in txt.lower():
                pages.append(i)
    return pages

# -------------------------
# DÉTECTION DES CLAUSES (prétraitement + segmentation)
# -------------------------
import spacy
nlp = spacy.load("fr_core_news_md")

def preprocess_sections(text: str) -> List[Dict]:
    """Découpe le texte en sections raisonnables (articles / paragraphes)"""
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # split heuristique
    parties = re.split(r'(?i)\n\s*(ARTICLE|SECTION|CHAPITRE|PARTIE)\b', text)
    # retourne des morceaux non vides
    clean = [p.strip() for p in parties if p and p.strip()]
    return [{"titre": f"sec_{i+1}", "contenu": p} for i, p in enumerate(clean)]

def extraire_phrases_prometteuses(text: str, min_len: int = 30) -> List[str]:
    """Retourne phrases candidates (SpaCy sentences) filtrées par taille"""
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]
    return sents

# -------------------------
# LLM: initialisation (local) & wrapper d'appel
# -------------------------
_llm_local_chain = None

def init_local_llm(model_id: str = LOCAL_MODEL_ID, device: str = DEVICE):
    """Charge un LLM local (transformers). Retourne pipeline de génération."""
    global _llm_local_chain
    if _llm_local_chain is not None:
        return _llm_local_chain
    logger.info(f"Chargement modèle local: {model_id} sur {device} (peut prendre du temps)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16 if device == "cuda" else None)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=1024)
    _llm_local_chain = gen
    return gen

def call_llm(prompt: str, backend: str = LLM_BACKEND, max_tokens: int = 1024) -> str:
    """
    Invokes the selected backend:
    - 'local' -> transformers pipeline (text-generation)
    - 'openai' -> (example) OpenAI ChatCompletion (user must configure)
    """
    logger.info("Appel LLM...")
    if backend == "local":
        gen = init_local_llm()
        out = gen(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
        return out
    elif backend == "openai":
        # Example: using OpenAI ChatCompletion
        # import openai
        # openai.api_key = os.environ.get("OPENAI_API_KEY")
        # resp = openai.ChatCompletion.create(
        #     model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=max_tokens
        # )
        # return resp["choices"][0]["message"]["content"]
        raise NotImplementedError("Mode 'openai' non implémenté dans cet exemple. Décommente et configure openai si besoin.")
    else:
        raise ValueError("Backend inconnu pour LLM")

# -------------------------
# PROMPT: compréhension et mapping champs -> valeurs
# -------------------------
PROMPT_TEMPLATE = """
Tu es un assistant qui comprend des cahiers des charges (appels d'offres) et qui doit
associer des champs d'une annexe (ou formulaire) aux valeurs correspondantes trouvées
dans les clauses administratives extraites.

Réponse attendue: **UNIQUE** objet JSON (aucun texte supplémentaire) de la forme:
{{
  "field_name_1": "valeur correspondante (ou null)",
  "field_name_2": "valeur correspondante (ou null)",
  ...
}}

Donne pour chaque champ:
- une valeur concise (si trouvée),
- null si aucune valeur n'est trouvée,
- évite d'inventer des valeurs.

Contrainte: si la valeur est une date, normalise en ISO (YYYY-MM-DD) quand possible.

Contexte : 
CLAUSES EXTRAIRES:
{clauses_context}

CHAMPS DU FORMULAIRE:
{fields_list}

Indique la source (courte) sous la forme "source":"extrait court" si possible, dans un champ séparé "meta", ex:
{{
  "field_name": "valeur",
  "_meta": { "field_name": "extrait court.." }
}}

Retourne uniquement le JSON.
"""

def build_prompt(clauses: Dict[str, List[str]], fields: List[str]) -> str:
    # clauses -> format textuel compact
    compact = []
    for cat, texts in clauses.items():
        compact.append(f"## {cat}\n" + "\n".join(texts[:20]))
    clauses_context = "\n\n".join(compact)[:15000]  # trim large context
    fields_list = json.dumps(fields, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(clauses_context=clauses_context, fields_list=fields_list)

# -------------------------
# UTIL: post-process mapping (validation, fallback heuristics)
# -------------------------
def validate_mapping(mapping: dict, fields: List[str]) -> dict:
    """
    S'assure que toutes les clés demandées sont présentes et que les valeurs sont strings or null.
    """
    out = {}
    for f in fields:
        val = mapping.get(f)
        if val is None:
            out[f] = None
        else:
            # protect against non-serialisable results
            if isinstance(val, (str, int, float)):
                out[f] = str(val)
            else:
                out[f] = str(val)
    return out

# -------------------------
# REMPLISSAGE PDF
# -------------------------
def remplir_acroform(pdf_in: str, pdf_out: str, mapping: Dict[str, Optional[str]]):
    """
    Remplit les champs AcroForm via pdfrw.
    mapping: {field_name: value or None}
    """
    pdf = PdfrwReader(pdf_in)
    if not hasattr(pdf, "Root") or not getattr(pdf.Root, "AcroForm", None):
        raise RuntimeError("PDF ne contient pas d'AcroForm")
    form = pdf.Root.AcroForm
    # ensure NeedAppearances
    form.update(PdfDict(NeedAppearances=PdfDict()))
    # iterate fields
    fields = form.Fields
    for f in fields:
        name = f.get("/T")
        if not name:
            continue
        key = name.strip("()")
        if key in mapping and mapping[key] is not None:
            f.update(PdfDict(V=str(mapping[key])))
    PdfrwWriter().write(pdf_out, pdf)
    logger.info(f"AcroForm rempli enregistré -> {pdf_out}")

def overlay_text_on_pages(pdf_in: str, pdf_out: str, mapping: Dict[str, Optional[str]], pages_to_fill: List[int]):
    """
    Pour PDF scanné: on ouvre les pages comme images, on dessine les paires 'field: value' en bas de page
    (ou sur pages annexes ciblées). C'est simple mais fonctionnel.
    """
    images = convert_from_path(pdf_in, poppler_path=POPPLER_PATH, dpi=200)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    new_pages = []
    for i, img in enumerate(images):
        im_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(im_rgb)
        if i in pages_to_fill:
            y = 50
            for k, v in mapping.items():
                txt = f"{k}: {v if v is not None else ''}"
                draw.text((50, y), txt, font=font)
                y += 18
                if y > im_rgb.height - 40:
                    break
        new_pages.append(im_rgb)
    if new_pages:
        new_pages[0].save(pdf_out, save_all=True, append_images=new_pages[1:])
        logger.info(f"PDF scanné rempli (overlay) -> {pdf_out}")

# -------------------------
# PIPELINE COMPLET
# -------------------------
def pipeline_extraction_et_remplissage(pdf_path: str, output_filled: str):
    # 1) extraction texte
    texte, meta = extraire_texte_intelligent(pdf_path)
    logger.info(f"Extraction ({meta['method']}) réalisée, {len(texte)} chars")

    # 2) détection clauses candidates (phrases)
    sections = preprocess_sections(texte)
    sentences = []
    for s in sections:
        sentences += extraire_phrases_prometteuses(s["contenu"])
    # group by simple heuristique: we keep top N sentences
    sentences = list(dict.fromkeys(sentences))[:300]  # dédupliquer et limiter

    # structure simple pour LLM
    clauses_struct = {"extraits": sentences}

    # 3) détecter champs AcroForm
    fields = detecter_champs_acroform(pdf_path)
    pages_annexes = detecter_pages_annexes(pdf_path)

    logger.info(f"Champs détectés (AcroForm): {fields}")
    logger.info(f"Pages annexes probables: {pages_annexes}")

    if not fields and not pages_annexes:
        logger.warning("Aucun champ AcroForm ni page 'annexe' repérée. Le pipeline procédera malgré tout à une suggestion de mapping.")

    # 4) appeler le LLM pour mapper
    prompt = build_prompt(clauses_struct, fields or [f"annexe_page_{p}" for p in pages_annexes] or ["champ_exemple"])
    llm_out = call_llm(prompt, backend=LLM_BACKEND)
    mapping = extract_json_from_text(llm_out) or {}
    mapping = validate_mapping(mapping, fields or [f"annexe_page_{p}" for p in pages_annexes] or ["champ_exemple"])

    # 5) remplissage
    try:
        if fields:
            remplir_acroform(pdf_path, output_filled, mapping)
        else:
            # overlay on annex pages
            overlay_text_on_pages(pdf_path, output_filled, mapping, pages_annexes if pages_annexes else [0])
    except Exception as e:
        logger.error(f"Erreur remplissage: {e}")
        # fallback: sauvegarder mapping en JSON
        with open("mapping_suggestion.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logger.info("Mapping sauvegardé dans mapping_suggestion.json (remplissage échoué)")

    # 6) sauvegarde clauses et mapping
    with open("clauses_extraites.json", "w", encoding="utf-8") as f:
        json.dump(clauses_struct, f, ensure_ascii=False, indent=2)
    with open("mapping_result.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logger.info("Pipeline terminé.")

# -------------------------
# MAIN (exécution)
# -------------------------
if __name__ == "__main__":
    pdf_input = "C:/Users/ASUS/Desktop/projet ia/testCC.pdf"
    pdf_output = "C:/Users/ASUS/Desktop/projet ia/annexe_remplie_llm.pdf"
    pipeline_extraction_et_remplissage(pdf_input, pdf_output)
