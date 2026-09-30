# Jan Sahayak AI 🤝

A smart AI-powered assistant that helps citizens understand Indian government schemes. Upload scheme PDFs, ask questions in plain language, and get accurate, objective answers backed by real document source citations.

Built with **Retrieval-Augmented Generation (RAG)** — every answer comes strictly from the uploaded scheme documents, preventing hallucinations.

---

## 🌟 Key Features

- 📄 **Upload & Index PDFs**: Parse official government scheme documents automatically into a vector database.
- 💬 **Grounded Q&A**: Ask natural language questions; receive answers strictly sourced from the indexed PDFs.
- 📍 **Source Verification**: View exact chunk content, document names, page numbers, and similarity confidence scores.
- 🎯 **Pre-loaded Demo Schemes**: Quick-start buttons for PMJDY, PM-KISAN, and Ayushman Bharat (PM-JAY).
- 🎨 **Modern Dark UI**: Premium glassmorphism dark theme designed with Streamlit.

---

## 🏗️ Architecture & Tech Stack

```
PDF Upload ➔ Text Extraction ➔ Overlapping Chunking ➔ Local ONNX Embeddings ➔ ChromaDB Vector Store
                                                                                     │
User Query  ➔ Semantic Search ➔ Top-K Retrieval ➔ Gemini 2.0 Flash Model ➔ Grounded Answer + Citations
```

- **Frontend**: Streamlit with custom CSS & dark theme
- **AI Model**: Google Gemini (via `langchain-google-genai`)
- **Embeddings**: Fast local CPU ONNX Embeddings (`all-MiniLM-L6-v2` via ChromaDB)
- **Vector DB**: ChromaDB (`langchain-chroma`)
- **Document Processing**: LangChain & PyPDF

---

## 🚀 Local Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Jan-Sahayak-AI.git
   cd Jan-Sahayak-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   - Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
   - Create a `.streamlit/secrets.toml` file or a `.env` file:
     ```toml
     GOOGLE_API_KEY = "your_actual_gemini_api_key"
     ```

4. **Generate Demo PDFs & Run App**:
   ```bash
   python generate_dummy_assets.py
   streamlit run app.py
   ```

---

## 🌐 Deploying to Streamlit Community Cloud (Step-by-Step)

Deploy your application online for free in 5 simple steps:

### Step 1: Push Code to GitHub
1. Create a new public or private repository on [GitHub](https://github.com/new) named `Jan-Sahayak-AI`.
2. Push your project code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Prepare Jan Sahayak AI for Streamlit Cloud deployment"
   git branch -M main
   git remote add origin https://github.com/your-username/Jan-Sahayak-AI.git
   git push -u origin main
   ```
   *(Note: Sensitive keys in `.env` and `.streamlit/secrets.toml` are automatically excluded by `.gitignore`)*

### Step 2: Log into Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **Continue with GitHub** to sign in with your GitHub account.

### Step 3: Create a New App
1. Click the **"New app"** button in the top right corner.
2. Select **"Use existing repo"**.
3. Fill in the deployment details:
   - **Repository**: `your-username/Jan-Sahayak-AI`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL slug if desired (e.g. `jan-sahayak-ai.streamlit.app`).

### Step 4: Add Gemini API Key to Secrets
1. Before clicking Deploy, click **"Advanced settings..."** (or go to **Settings ➔ Secrets** after creating).
2. Under the **Secrets** text area, paste your Google API key in TOML format:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
   ```
3. Click **Save**.

### Step 5: Deploy & Access App
1. Click **Deploy!**
2. Streamlit Cloud will automatically build the environment using `requirements.txt` and launch your app.
3. Your live application will be accessible worldwide! Any new git commits pushed to `main` will automatically update the live app.

---

## 📂 Project Structure

```
Jan-Sahayak-AI/
├── app.py                      # Main Streamlit web application & UI layout
├── chatbot.py                  # Grounded RAG query logic & Gemini integration
├── vector_store.py             # ChromaDB vector store management & ONNX embeddings
├── pdf_processor.py            # PDF document parsing & text chunking
├── utils.py                    # File handling & validation helpers
├── generate_dummy_assets.py    # Script to create sample government scheme PDFs
├── requirements.txt            # Production dependencies for Streamlit Cloud
├── .gitignore                  # Keeps secrets & temporary database out of git
└── .streamlit/
    ├── config.toml             # Custom UI theme settings
    └── secrets.toml.example    # Secrets template for cloud deployment
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
