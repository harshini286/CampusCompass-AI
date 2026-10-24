
# 🎓 CampusCompass AI

An AI-powered campus knowledge assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from college documents.

## Features
- Upload and extract text from PDF documents
- Semantic search using Sentence Transformers
- Vector storage and retrieval with ChromaDB
- Gemini-powered answers grounded in retrieved content
- Source document and page references
- Placement eligibility checker
- Cross-document comparison for placement notices

## Tech Stack
- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- PyPDF

## Run Locally

1. Clone or download this repository.
2. Create and activate a virtual environment.
3. Install the required dependencies.
4. Add your Gemini API key to a `.env` file.
5. Run the application.

```bash
python -m pip install streamlit google-genai python-dotenv pypdf chromadb sentence-transformers
streamlit run app.py
```

Add the following to `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

Never upload your real API key or `.env` file to GitHub.

## Disclaimer
Placement eligibility uses fictional demo rules and must not be treated as official college or employer policy.
