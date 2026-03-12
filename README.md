# gocalma-shield
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![Presidio](https://img.shields.io/badge/NLP-Microsoft_Presidio-0078D4.svg)](https://microsoft.github.io/presidio/)

> Built for the **GenAI Zürich Hackathon 2026** - *The GoCalma Challenge (AI-Powered Privacy Redaction)*.

## The Problem
Every day, professionals upload sensitive documents containing PII (Personally Identifiable Information) to AI tools like ChatGPT or Claude, losing control over their personal data. To use AI safely, data must be redacted **before** it ever leaves the user's device. 

## Our Solution
**GoCalma Shield** is a 100% local, lightweight middleware designed to neutralize privacy risks. It automatically identifies and masks sensitive information using offline NLP models, ensuring that the cloud LLMs only receive sanitized, format-preserved data for reasoning.

### Key Features
1. **100% Local Processing**: Powered by Microsoft `presidio-analyzer` and offline `spaCy` large models. No cloud API is used for anonymization. Your raw data never touches the internet.
2. **Dual Input Modes**:
   - **File Mode**: Batch process multiple files (TXT, PDF, CSV, JSON) at once. The app auto-redacts upon upload.
   - **Text Mode**: A simple text area for quick copy-paste operations.
3. **Multi-File Handling**: Upload and process a collection of documents simultaneously. The app provides both individual and combined redacted outputs for download.
4. **Format-Preserving Masking**: Smart masking for Phones, Emails, IDs, Names, and Addresses. The output remains highly readable for LLMs to understand the context.
5. **Dual-Track AI Interaction**:
   - **API Geek Mode**: Bring your own Key (BYOK) to stream LLM responses directly within the app.
   - **Web Beginner Mode**: One-click copy and quick-launch buttons to native ChatGPT/Claude web interfaces.

## Tech Stack
- **Frontend & Backend**: Python + Streamlit
- **NLP Engine**: Microsoft Presidio, spaCy (`en_core_web_lg`, `zh_core_web_lg`)
- **Document Parser**: PyPDF (for PDF extraction)
- **LLM Integration**: OpenAI-compatible streaming API

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Download Offline NLP Models
To ensure absolute privacy, the app requires local language models:
code
Bash
python -m spacy download en_core_web_lg
python -m spacy download zh_core_web_lg
3. Start the Application
code
Bash
streamlit run app.py
### How it Works
Choose Mode: Select "File" for batch processing or "Text" for a single input.
Upload/Paste: Drag and drop multiple files or paste text on the left panel.
Auto-Shield: The app instantly redacts PII offline. In File Mode, this happens automatically upon upload. In Text Mode, click the desensitize button.
Interact & Download: Use the right panel to safely prompt your favorite LLM. In the middle panel, download individual or combined redacted files.
Own your data. Empower your AI.
