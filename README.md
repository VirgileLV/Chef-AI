# 👨‍🍳 Chef-AI – Your AI Cooking Assistant

Chef-AI is an AI-powered cooking assistant that allows users to upload recipe documents (CSV or PDF), ask natural language questions, and receive accurate answers grounded in your uploaded content.

It combines **Groq's LLM**, **Hugging Face Embeddings**, and **FAISS vector search**, all wrapped inside an intuitive **Streamlit interface**. The app uses a **Retrieval-Augmented Generation (RAG)** architecture to ensure high-quality, context-aware answers.

---

## 🚀 Features

- 📄 Upload recipes in CSV or PDF format
- 🔍 Extract and chunk cooking instructions, ingredients, and nutrition info
- 🤖 Use Groq’s high-performance LLM for response generation
- 🧠 Smart context builder for token-efficient prompts
- ✅ Answers backed by source documents
- 🪄 JSON-compatible prompt templates with guaranteed structure
- 🧾 Automatic indexing and deduplication via file modification detection

---

## 📂 Project Structure

```
ChefFriends/
├── app.py                  # Streamlit UI frontend
├── data_pipeline.py        # Core pipeline for ingestion, vector search & LLM setup
├── cache/
│   ├── indexed_files.json  # Tracks processed files and timestamps
|   └──  faiss_index/
│       ├── index.faiss         # FAISS vector index
│       └── index.pkl           # FAISS metadata (docstore)
├── data/
│   ├── recipes_csv/        # Uploaded CSV recipes
│   └── recipes_pdf/        # Uploaded PDF recipes
├── requirements.txt        # Python dependencies
└── .env                    # API keys for Groq and Hugging Face
```

---

## 🧪 Installation

1. Clone the repository:

```bash
cd ChefFriends
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following:

```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

> You can get a Groq API key from https://console.groq.com and Hugging Face from https://huggingface.co/settings/tokens

---

## 💡 Usage

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Then:
- Upload `.csv` or `.pdf` recipe files in the sidebar
- Ask cooking-related questions in the input field
- Get AI-generated responses along with source document previews

---

## 🧠 Modules Explained

### `data_pipeline.py`

This module handles **data ingestion**, **preprocessing**, **embedding**, **indexing**, and **LLM configuration**:

#### Key Components:

- **`load_from_csv(path)`**  
  Loads CSV files, extracts titles, ingredients, and instructions, maps optional metadata (nutrition, cuisine, etc.), and wraps them in LangChain `Document` objects.

- **`load_from_pdf(path)`**  
  Uses `pdfplumber` to extract page-level text content from PDF files and turn them into LangChain documents.

- **`detect_and_process_new_files(csv_folder, pdf_folder)`**  
  Detects new or modified files by comparing timestamps against a cache, loads content, and splits it into text chunks using `RecursiveCharacterTextSplitter`.

- **`initialize_vectorstore()`**  
  Loads or initializes a local FAISS vector index, using `sentence-transformers/all-mpnet-base-v2` embeddings via Hugging Face.

- **`update_vectorstore()`**  
  Adds newly processed document chunks to FAISS and saves the updated index and docstore.

- **`build_context_smart(docs, query, max_tokens)`**  
  Token-aware function that builds a limited-size prompt context using the Hugging Face tokenizer.

- **`truncate_prompt(prompt, tokenizer, max_tokens)`**  
  Ensures the final string prompt fits within the model’s token budget.

- **`load_llm()`**  
  Initializes a `ChatGroq` model (e.g., `llama-3.3-70b-versatile`) and wraps it in a `ChatPromptTemplate` chain for structured Q&A.

---

### `app.py`

This is the **user interface** powered by **Streamlit**. It ties everything together:

- Provides file uploader for `.csv` and `.pdf`
- Displays user input field and model-generated response
- Renders relevant source documents from which the answer was generated
- Uses caching to avoid reloading models or reprocessing data unnecessarily

---

## 📤 Input File Requirements

### CSV format

The CSV should contain at least these columns (case-insensitive):

- `name` or `title`
- `ingredients`
- `steps` or `instructions`

Optionally, it may contain:
- `cook_time`, `nutrition_score`, `cuisine`, `diet`

### PDF format

PDFs should be structured with clearly readable recipe content (title, ingredients, steps).

---

## 🧠 AI Stack

| Component        | Technology                                 |
|------------------|--------------------------------------------|
| LLM              | Groq (`llama-3.3-70b-versatile`)           |
| Embeddings       | `sentence-transformers/all-mpnet-base-v2`  |
| Vector DB        | FAISS                                      |
| Prompt Framework | LangChain (`ChatPromptTemplate`)           |
| Output Parser    | Optional: `JsonOutputParser`               |

---

## 🛠 Future Improvements

- 🧾 Add image OCR for handwritten recipes
- 🥗 Filter queries based on diet/nutrition constraints
- 🔍 Add advanced keyword search alongside RAG
- 📦 Dockerize for deployment

---

## 📣 Example Questions

- “What can I cook with lentils and carrots?”
- “How long does the mushroom risotto take?”
- “Give me a healthy dinner under 30 minutes”
- “Which of these recipes is gluten-free?”

---

## 🤝 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
