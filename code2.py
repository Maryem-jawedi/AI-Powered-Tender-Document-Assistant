import fitz
import os
import pytesseract
from pdf2image import convert_from_path
import pdfplumber 
import re 
import spacy
from collections import defaultdict
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pdfrw import PdfReader, PdfWriter, PdfDict
from PIL import ImageDraw, ImageFont
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClauseAdministrative:
    """Structure pour représenter une clause administrative"""
    type: str
    texte: str
    section: str
    confiance: float
    page: Optional[int] = None
    urgence: Optional[str] = None

@dataclass
class ConfigAO:
    """Configuration pour l'assistant AO"""
    model_name: str = "facebook/bart-large-mnli"
    poppler_path: str = r"C:\Users\ASUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
    spacy_model: str = "fr_core_news_md"
    confidence_threshold: float = 0.7
    min_text_length: int = 20

class AssistantAO:
    """Classe principale pour l'assistant IA d'appels d'offres"""
    
    def __init__(self, config: ConfigAO = None):
        self.config = config or ConfigAO()
        self._init_models()
        
    def _init_models(self):
        """Initialise les modèles IA et NLP"""
        try:
            logger.info("Initialisation des modèles...")
            
            # Chargement du modèle BART
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, force_download=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, resume_download=False
            )
            
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Chargement de spaCy
            self.nlp = spacy.load(self.config.spacy_model)
            
            logger.info("✅ Modèles initialisés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des modèles: {e}")
            raise
    
    # Catégories étendues et plus précises
    CATEGORIES = [
        "document obligatoire",
        "attestation requise", 
        "délai contractuel",
        "condition juridique",
        "critère d'évaluation",
        "condition technique",
        "garantie financière",
        "assurance obligatoire",
        "certification requise",
        "déclaration sur l'honneur"
    ]

    KEYWORDS_EXTENDED = {
        "documents_obligatoires": [
            "dc1", "dc2", "dc3", "dc4", "dume", 
            "kbis", "extrait", "attestation fiscale", "urssaf",
            "joindre", "fournir", "obligatoire", "requis"
        ],
        "attestations": [
            "attestation", "certificat", "agrément", "habilitation",
            "qualification", "certification", "accréditation"
        ],
        "delais": [
            "délai", "échéance", "date limite", "jours ouvrables", 
            "calendaires", "avant le", "au plus tard"
        ],
        "juridique": [
            "conformité", "légal", "réglementaire", "loi", "article",
            "décret", "arrêté", "code", "jurisprudence"
        ],
        "financier": [
            "caution", "garantie", "nantissement", "chiffre d'affaires",
            "bilan", "compte de résultat", "solvabilité"
        ],
        "technique": [
            "norme", "spécification", "performance", "essai",
            "test", "validation", "homologation"
        ]
    }

    def extraire_texte_intelligent(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extraction intelligente avec fallback OCR et métadonnées"""
        metadata = {"method": "direct", "pages": 0, "has_forms": False}
        
        try:
            # Tentative d'extraction directe
            texte = ""
            with fitz.open(pdf_path) as doc:
                metadata["pages"] = len(doc)
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    texte += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text
                    
                    # Vérification des formulaires
                    if page.first_widget:
                        metadata["has_forms"] = True
                        
            # Si le texte est trop court, utiliser OCR
            if len(texte.strip()) < 100:
                logger.warning("Texte extrait trop court, utilisation de l'OCR...")
                texte = self._extraire_texte_ocr(pdf_path)
                metadata["method"] = "ocr"
                
            return texte, metadata
            
        except Exception as e:
            logger.error(f"Erreur extraction directe: {e}")
            texte = self._extraire_texte_ocr(pdf_path)
            metadata["method"] = "ocr_fallback"
            return texte, metadata

    def _extraire_texte_ocr(self, pdf_path: str) -> str:
        """Extraction OCR optimisée"""
        try:
            images = convert_from_path(pdf_path, poppler_path=self.config.poppler_path, dpi=300)
            texte_total = ""
            
            for i, image in enumerate(images):
                # Amélioration de la qualité pour OCR
                text = pytesseract.image_to_string(
                    image, 
                    lang='fra',
                    config='--psm 6 --oem 3'  # Meilleure configuration OCR
                )
                texte_total += f"\n\n--- Page {i+1} ---\n\n" + text
                
            return texte_total
        except Exception as e:
            raise Exception(f"Erreur OCR: {str(e)}")

    def preprocess_text_avance(self, text: str) -> List[Dict]:
        """Préprocessing avancé avec segmentation intelligente"""
        
        # Nettoyage du texte
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Suppression numéros de page
        text = re.sub(r'\s+', ' ', text)  # Normalisation espaces
        
        # Segmentation par sections
        patterns_sections = [
            r'\n\s*(ARTICLE|SECTION|CHAPITRE|PARTIE|TITRE)\s+[IVXLCDM0-9]+\s*[\-\–]?\s*(.+?)(?=\n|$)',
            r'\n\s*([0-9]+\.?[0-9]*)\s+[\-\–]?\s*(.+?)(?=\n\s*[0-9]+\.|\Z)',
            r'\n\s*([A-Z][\.)])\s+(.+?)(?=\n\s*[A-Z][\.)]|\Z)'
        ]
        
        sections = []
        for pattern in patterns_sections:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                sections.append({
                    'titre': match.group(1) + (match.group(2) if len(match.groups()) > 1 else ''),
                    'contenu': match.group(0),
                    'position': match.start()
                })
        
        # Si pas de sections détectées, diviser par paragraphes
        if not sections:
            paragraphes = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            sections = [{'titre': f'Paragraphe {i+1}', 'contenu': p, 'position': i} 
                       for i, p in enumerate(paragraphes)]
        
        return sections

    def detecter_clauses_avancees(self, text: str) -> List[ClauseAdministrative]:
        """Détection avancée des clauses avec analyse contextuelle"""
        sections = self.preprocess_text_avance(text)
        clauses_detectees = []
        
        for section in sections:
            doc = self.nlp(section['contenu'])
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) < self.config.min_text_length:
                    continue
                
                # Score basé sur les mots-clés
                scores_categories = {}
                for cat, keywords in self.KEYWORDS_EXTENDED.items():
                    score = sum(1 for keyword in keywords 
                              if keyword.lower() in sent_text.lower()) / len(keywords)
                    if score > 0:
                        scores_categories[cat] = score
                
                # Classification avec transformers pour les phrases prometteuses
                if scores_categories or any(word in sent_text.lower() 
                                         for word in ['obligatoire', 'requis', 'fournir', 'joindre']):
                    
                    try:
                        result = self.classifier(
                            sent_text,
                            candidate_labels=self.CATEGORIES,
                            hypothesis_template="Cette phrase concerne {}"
                        )
                        
                        if result['scores'][0] >= self.config.confidence_threshold:
                            # Détection d'urgence
                            urgence = self._evaluer_urgence(sent_text)
                            
                            clause = ClauseAdministrative(
                                type=result['labels'][0],
                                texte=sent_text,
                                section=section['titre'],
                                confiance=round(result['scores'][0], 3),
                                urgence=urgence
                            )
                            clauses_detectees.append(clause)
                            
                    except Exception as e:
                        logger.warning(f"Erreur classification pour: {sent_text[:50]}... - {e}")
        
        # Déduplication et tri
        clauses_uniques = self._dedupliquer_clauses(clauses_detectees)
        return sorted(clauses_uniques, key=lambda x: x.confiance, reverse=True)

    def _evaluer_urgence(self, texte: str) -> str:
        """Évalue le niveau d'urgence d'une clause"""
        mots_urgents = ['immédiat', 'urgent', 'avant le', 'au plus tard', 'délai impératif']
        mots_delai_court = ['24h', '48h', '72h', 'sous 3 jours', 'avant vendredi']
        
        texte_lower = texte.lower()
        
        if any(mot in texte_lower for mot in mots_urgents):
            return "haute"
        elif any(mot in texte_lower for mot in mots_delai_court):
            return "moyenne" 
        elif 'délai' in texte_lower or 'échéance' in texte_lower:
            return "normale"
        else:
            return "faible"

    def _dedupliquer_clauses(self, clauses: List[ClauseAdministrative]) -> List[ClauseAdministrative]:
        """Supprime les clauses en double basé sur la similarité du texte"""
        clauses_uniques = []
        
        for clause in clauses:
            est_doublon = False
            for unique in clauses_uniques:
                # Calcul de similarité simple (Jaccard)
                mots_clause = set(clause.texte.lower().split())
                mots_unique = set(unique.texte.lower().split())
                
                intersection = len(mots_clause.intersection(mots_unique))
                union = len(mots_clause.union(mots_unique))
                
                similarite = intersection / union if union > 0 else 0
                
                if similarite > 0.8:  # 80% de similarité = doublon
                    est_doublon = True
                    # Garder celui avec la meilleure confiance
                    if clause.confiance > unique.confiance:
                        clauses_uniques.remove(unique)
                        clauses_uniques.append(clause)
                    break
            
            if not est_doublon:
                clauses_uniques.append(clause)
        
        return clauses_uniques

    def generer_rapport_analyse(self, clauses: List[ClauseAdministrative], 
                               metadata: Dict, pdf_path: str) -> Dict:
        """Génère un rapport complet d'analyse"""
        
        # Statistiques par type
        stats_types = defaultdict(int)
        for clause in clauses:
            stats_types[clause.type] += 1
        
        # Analyse d'urgence
        stats_urgence = defaultdict(int)
        for clause in clauses:
            stats_urgence[clause.urgence or 'non_evaluee'] += 1
        
        # Détection des documents manquants potentiels
        documents_standards = ['dc1', 'dc2', 'kbis', 'attestation urssaf', 'attestation fiscale']
        documents_detectes = []
        for clause in clauses:
            for doc in documents_standards:
                if doc in clause.texte.lower():
                    documents_detectes.append(doc)
        
        documents_potentiellement_manquants = [
            doc for doc in documents_standards if doc not in documents_detectes
        ]
        
        rapport = {
            "fichier_analyse": os.path.basename(pdf_path),
            "metadata": metadata,
            "resume": {
                "total_clauses": len(clauses),
                "confiance_moyenne": round(sum(c.confiance for c in clauses) / len(clauses), 2) if clauses else 0,
                "types_detectes": len(stats_types),
                "urgences_detectees": len([c for c in clauses if c.urgence in ['haute', 'moyenne']])
            },
            "statistiques": {
                "par_type": dict(stats_types),
                "par_urgence": dict(stats_urgence)
            },
            "alertes": {
                "documents_potentiellement_manquants": documents_potentiellement_manquants,
                "clauses_haute_urgence": [c.texte for c in clauses if c.urgence == 'haute']
            },
            "recommendations": self._generer_recommendations(clauses, stats_types)
        }
        
        return rapport

    def _generer_recommendations(self, clauses: List[ClauseAdministrative], 
                                stats_types: Dict) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        if stats_types.get('délai contractuel', 0) > 0:
            recommendations.append("⏰ Vérifiez immédiatement tous les délais mentionnés")
        
        if stats_types.get('document obligatoire', 0) > 5:
            recommendations.append("📋 Nombreux documents obligatoires détectés - créez une checklist")
        
        if stats_types.get('garantie financière', 0) > 0:
            recommendations.append("💰 Garanties financières requises - contactez votre banque")
        
        if any(c.urgence == 'haute' for c in clauses):
            recommendations.append("🚨 Clauses haute urgence détectées - action immédiate requise")
        
        if not recommendations:
            recommendations.append("✅ Analyse terminée - aucune alerte critique détectée")
        
        return recommendations

    def sauvegarder_analyse(self, clauses: List[ClauseAdministrative], 
                           rapport: Dict, output_dir: str = "output"):
        """Sauvegarde l'analyse dans plusieurs formats"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Format JSON structuré
        clauses_dict = {}
        for clause in clauses:
            if clause.type not in clauses_dict:
                clauses_dict[clause.type] = []
            clauses_dict[clause.type].append({
                'texte': clause.texte,
                'section': clause.section,
                'confiance': clause.confiance,
                'urgence': clause.urgence
            })
        
        with open(f"{output_dir}/clauses_structurees.json", "w", encoding="utf-8") as f:
            json.dump(clauses_dict, f, ensure_ascii=False, indent=4)
        
        # Rapport complet
        with open(f"{output_dir}/rapport_analyse.json", "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=4)
        
        # Rapport lisible
        self._generer_rapport_lisible(rapport, f"{output_dir}/rapport_analyse.md")
        
        logger.info(f"✅ Analyse sauvegardée dans {output_dir}/")

    def _generer_rapport_lisible(self, rapport: Dict, filepath: str):
        """Génère un rapport en markdown lisible"""
        
        contenu = f"""# Rapport d'Analyse - {rapport['fichier_analyse']}

## 📊 Résumé

- **Total clauses détectées**: {rapport['resume']['total_clauses']}
- **Confiance moyenne**: {rapport['resume']['confiance_moyenne']}
- **Types différents**: {rapport['resume']['types_detectes']}
- **Clauses urgentes**: {rapport['resume']['urgences_detectees']}

## 📈 Statistiques détaillées

### Par type de clause
"""
        
        for type_clause, count in rapport['statistiques']['par_type'].items():
            contenu += f"- **{type_clause}**: {count} clause(s)\n"
        
        contenu += "\n### Par niveau d'urgence\n"
        for urgence, count in rapport['statistiques']['par_urgence'].items():
            contenu += f"- **{urgence}**: {count} clause(s)\n"
        
        contenu += "\n## 🚨 Alertes\n"
        
        if rapport['alertes']['documents_potentiellement_manquants']:
            contenu += "\n### Documents potentiellement manquants\n"
            for doc in rapport['alertes']['documents_potentiellement_manquants']:
                contenu += f"- {doc}\n"
        
        if rapport['alertes']['clauses_haute_urgence']:
            contenu += "\n### Clauses haute urgence\n"
            for clause in rapport['alertes']['clauses_haute_urgence']:
                contenu += f"- {clause[:100]}...\n"
        
        contenu += "\n## 💡 Recommandations\n"
        for rec in rapport['recommendations']:
            contenu += f"- {rec}\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(contenu)

    def analyser_appel_offre(self, pdf_path: str, output_dir: str = "output") -> Dict:
        """Pipeline complet d'analyse d'un appel d'offres"""
        
        logger.info(f"🔍 Début d'analyse de: {pdf_path}")
        
        # 1. Extraction de texte
        texte, metadata = self.extraire_texte_intelligent(pdf_path)
        logger.info(f"📄 Texte extrait ({metadata['method']}): {len(texte)} caractères")
        
        # 2. Détection des clauses
        clauses = self.detecter_clauses_avancees(texte)
        logger.info(f"🔍 {len(clauses)} clauses administratives détectées")
        
        # 3. Génération du rapport
        rapport = self.generer_rapport_analyse(clauses, metadata, pdf_path)
        
        # 4. Sauvegarde
        self.sauvegarder_analyse(clauses, rapport, output_dir)
        
        # 5. Affichage résumé console
        print(f"\n{'='*80}")
        print(f"✅ ANALYSE TERMINÉE - {rapport['fichier_analyse']}")
        print(f"{'='*80}")
        print(f"📊 {rapport['resume']['total_clauses']} clauses détectées")
        print(f"🎯 Confiance moyenne: {rapport['resume']['confiance_moyenne']}")
        print(f"⚠️  {rapport['resume']['urgences_detectees']} clause(s) urgente(s)")
        print(f"\n💡 Recommandations principales:")
        for rec in rapport['recommendations'][:3]:
            print(f"   {rec}")
        print(f"\n📁 Résultats sauvegardés dans: {output_dir}/")
        
        return rapport


# Fonction principale améliorée
def main():
    """Fonction principale avec gestion des erreurs et interface utilisateur"""
    
    print("🤖 Assistant IA pour Appels d'Offres - Version Avancée")
    print("="*60)
    
    try:
        # Configuration
        config = ConfigAO()
        assistant = AssistantAO(config)
        
        # Chemin du PDF (à adapter)
        chemin_pdf = input("\n📁 Chemin vers le cahier des charges (ou Entrée pour test): ").strip()
        if not chemin_pdf:
            chemin_pdf = "C:/Users/ASUS/Desktop/projet ia/testCC.pdf"
        
        if not os.path.exists(chemin_pdf):
            print(f"❌ Fichier non trouvé: {chemin_pdf}")
            return
        
        # Analyse complète
        rapport = assistant.analyser_appel_offre(chemin_pdf)
        
        # Options post-analyse
        print(f"\n🔧 Options disponibles:")
        print("1. Afficher les clauses détaillées")
        print("2. Générer des documents types")
        print("3. Créer une checklist")
        print("4. Quitter")
        
        choix = input("\nVotre choix (1-4): ").strip()
        
        if choix == "1":
            # Charger et afficher les clauses
            with open("output/clauses_structurees.json", "r", encoding="utf-8") as f:
                clauses = json.load(f)
            
            for type_clause, liste_clauses in clauses.items():
                print(f"\n📋 {type_clause.upper()} ({len(liste_clauses)} clause(s)):")
                for i, clause in enumerate(liste_clauses[:3], 1):  # Max 3 par type
                    print(f"   {i}. {clause['texte'][:100]}...")
                    print(f"      Confiance: {clause['confiance']} | Urgence: {clause['urgence']}")
        
        elif choix == "2":
            print("🔨 Génération de documents types - À implémenter")
            
        elif choix == "3":
            print("📝 Création de checklist - À implémenter")
        
        print("\n✅ Session terminée avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analyse interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        print(f"❌ Une erreur est survenue: {e}")

if __name__ == "__main__":
    main()