# Medimind AI Chatbot

A simple Python Retrieval-Augmented Generation chatbot for healthcare education.

The app retrieves relevant information from local text files in `data/health_docs`
and uses that context to answer questions. It can run in two modes:

- Retrieval-only fallback: returns a grounded answer from the best matching documents.
- Optional Ollama generation: sends the retrieved context to a local Ollama model.

> This project is for educational use only. It is not a medical device and should
> not be used for diagnosis, treatment, or emergency care.

## Features

- Streamlit chat interface
- Local healthcare knowledge base
- TF-IDF document retrieval with scikit-learn
- Source snippets shown with each answer
- Optional local LLM through Ollama
- Safety prompt for healthcare boundaries

## Project Structure

```text
healthcare-rag-chatbot/
  app.py
  rag.py
  ollama_client.py
  requirements.txt
  data/
    health_docs/
      diabetes.txt
      hypertension.txt
      first_aid.txt
```

## Setup

```powershell
cd medimind-ai-chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Optional Ollama Mode

Install and run Ollama, then pull a model:

```powershell
ollama pull llama3.1
ollama serve
```

In another terminal:

```powershell
$env:OLLAMA_MODEL="llama3.1"
streamlit run app.py
```

If Ollama is not available, the app automatically uses retrieval-only answers.

## Add Your Own Documents

Add `.txt` files to `data/health_docs`, then restart the Streamlit app.
Use trusted, reviewed content and avoid storing personal health information.
