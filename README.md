# AI-Powered Tender Document Assistant

An intelligent assistant designed to automate the preparation of administrative documents for tender applications. The system analyzes tender specifications, extracts administrative and legal requirements, and automatically fills in the required forms and annexes.

## Overview

Responding to a call for tenders often involves analyzing lengthy specifications, collecting information from multiple documents, and manually completing administrative forms. This process can be time-consuming and prone to errors.

This project aims to simplify this process through Artificial Intelligence, Natural Language Processing (NLP), and document-processing techniques.

The assistant processes PDF tender specifications, identifies relevant requirements, extracts structured information, and generates pre-filled administrative documents ready for review.

## Key Features

- **Intelligent PDF Processing:** Extracts text from both digitally generated and scanned PDF documents.
- **OCR Integration:** Uses Optical Character Recognition to process scanned documents and image-based PDFs.
- **Administrative Requirements Extraction:** Identifies and classifies administrative, legal, and contractual requirements using NLP and LLMs.
- **Automatic Form Detection:** Detects fields in interactive PDF forms and identifies fillable areas in scanned documents.
- **Intelligent Form Filling:** Automatically maps extracted information to the corresponding fields in administrative forms.
- **Data Validation:** Detects missing fields, inconsistencies, and incorrect data formats.
- **Document Generation:** Produces completed PDF documents while preserving their original layout.
- **Critical Points Detection:** Highlights mandatory documents, signatures, deadlines, and other important requirements.
- **Interactive Web Interface:** Provides a user-friendly interface for uploading documents, reviewing results, and downloading generated files.

## System Architecture

The solution follows a modular processing pipeline:

1. **PDF Text Extraction**
   - PyMuPDF for digitally generated PDFs.
   - pdf2image and Tesseract OCR for scanned PDFs.

2. **Text Preprocessing and Segmentation**
   - Text cleaning and normalization.
   - Removal of page numbers and unnecessary elements.
   - Segmentation into logical sections.

3. **Linguistic Analysis**
   - spaCy (`fr_core_news_md`) for linguistic processing and entity recognition.
   - Identification of important administrative information.

4. **AI-Powered Requirements Extraction**
   - OpenAI GPT-4o-mini API.
   - Extraction of administrative and legal clauses.
   - Structuring extracted information into JSON.

5. **PDF Form Field Detection**
   - PyPDF2 for interactive PDF forms (AcroForm).
   - Tesseract OCR for detecting fields in scanned documents.

6. **Intelligent Form Filling**
   - Heuristic matching using RapidFuzz.
   - AI-assisted mapping between extracted information and form fields.

7. **Validation and Document Generation**
   - Verification of mandatory fields and data consistency.
   - PDF modification and generation using PyMuPDF.

## Processing Workflow

```text
       Tender Specifications (PDF)
                    |
                    v
          PDF Text Extraction
             (PyMuPDF / OCR)
                    |
                    v
       Text Preprocessing & Segmentation
                    |
                    v
       Administrative Requirements
              Extraction (NLP)
                    |
                    v
         AI-Based Information Structuring
               (GPT-4o-mini)
                    |
                    v
          PDF Form Field Detection
                    |
                    v
          Intelligent Field Mapping
           (RapidFuzz + AI)
                    |
                    v
          Validation & Consistency
                 Checking
                    |
                    v
          Completed Administrative
                 Documents
                    |
                    v
          PDF Generation & Download
```

## Input and Output

### Inputs

- Tender specifications (PDF files).
- Supporting documents and administrative form templates.
- Company information, imported or entered manually.

### Outputs

- Pre-filled administrative documents.
- Completed PDF or Word files, depending on the required format.
- A list of critical points requiring verification, such as missing documents, signatures, and mandatory information.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| PDF Processing | PyMuPDF, PyPDF2 |
| OCR | Tesseract, pytesseract, pdf2image |
| NLP | spaCy |
| Generative AI | OpenAI GPT-4o-mini API |
| Fuzzy Matching | RapidFuzz |
| Document Generation | ReportLab, PyMuPDF |
| User Interface | Gradio |
| Version Control | Git, GitHub |
| Project Management | Trello |

## User Interface

The application provides a web-based interface built with Gradio.

Its main capabilities include:

- Uploading tender specifications in PDF format.
- Automatically analyzing the uploaded documents.
- Generating and filling administrative forms and annexes.
- Displaying warnings and recommendations when missing or inconsistent information is detected.
- Downloading the generated documents.

## Installation and Setup

### Prerequisites

- Python 3.11
- Tesseract OCR
- Git
- An OpenAI API key

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the required libraries according to the project's dependency configuration.

### 4. Configure the OpenAI API

Configure your OpenAI API key using the environment-variable or configuration mechanism implemented in the project.

**Important:** Keep API keys and other sensitive credentials out of the repository.

### 5. Launch the application

Run the project's Gradio entry point. The exact command depends on the application entry-point file in the repository.

## Testing and Validation

The assistant was tested using different types of PDF documents:

- Digitally generated PDFs containing selectable text.
- Scanned PDFs requiring OCR processing.
- Interactive PDF forms containing AcroForm fields.

The validation process focused on:

- Accuracy of text extraction.
- Detection of administrative and legal requirements.
- Correct identification of form fields.
- Automatic completion of administrative forms.
- Detection of missing information and inconsistencies.
- Preservation of the original PDF layout.

## Development Methodology

The project was developed using an **Agile Scrum methodology**, with iterative development cycles (sprints).

The implementation was progressively structured around the following milestones:

1. PDF text extraction and OCR integration.
2. Administrative field detection.
3. AI-based extraction of administrative clauses.
4. Intelligent form filling.
5. Validation and final document generation.

## Expected Benefits

- Reduced time spent preparing tender applications.
- Less repetitive manual data entry.
- Improved consistency and reliability of administrative documents.
- Early identification of missing information and potential compliance issues.
- Reusability of company information across future tender applications.

## Future Improvements

Potential extensions include:

- Supporting additional document formats and form templates.
- Improving extraction accuracy for complex legal and administrative clauses.
- Extending validation rules for different tender requirements.
- Enhancing the review process with human-in-the-loop validation.
- Improving document generation and formatting compatibility.

## Author

**Maryem Jawedi**

Engineering Student – National School of Computer Science (ENSI)

**Internship Company:** Eventizer Platforms  
**Academic Year:** 2024–2025

## Acknowledgments

Special thanks to Eventizer Platforms and my professional supervisor, **Yessine Abess**, for their guidance and support throughout this internship.

---

*This project was developed as part of a summer internship at Eventizer Platforms.*
