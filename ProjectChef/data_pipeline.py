import pdfplumber
import pandas as pd
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
from transformers import AutoTokenizer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TITLE_COLUMNS = ["name", "title", "recipe_name", "dish"]
INGREDIENT_COLUMNS = ["ingredients", "ingr", "components"]
STEPS_COLUMNS = ["steps", "instructions", "directions", "method"]

OPTIONAL_FIELDS = {
    "cook_time": ["cook_time", "cooking_time", "ready_in", "prep_time", "minutes"],
    "nutrition_score": ["nutrition_score", "nutrition", "health_score"],
    "cuisine": ["cuisine", "category", "region", "state"],
    "diet": ["diet", "dietary_restrictions", "allergies"],
}

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "indexed_files.json"
FAISS_INDEX_FILE = CACHE_DIR / "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def normalize_column(columns, alias_pool):
    for col in columns:
        if col.lower() in alias_pool:
            return col
    return None

def load_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(Document(page_content=text, metadata={"source": f"{path.name}-page-{i}"}))
    return pages

def load_from_csv(path, max_rows=200):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    title_col = normalize_column(df.columns, TITLE_COLUMNS)
    ingredients_col = normalize_column(df.columns, INGREDIENT_COLUMNS)
    steps_col = normalize_column(df.columns, STEPS_COLUMNS)

    if not (title_col and ingredients_col and steps_col):
        raise ValueError(f"Missing required fields in: {path}")

    df = df.dropna(subset=[title_col, ingredients_col, steps_col]).head(max_rows)

    optional_mapping = {}
    for field_name, aliases in OPTIONAL_FIELDS.items():
        found_col = normalize_column(df.columns, aliases)
        optional_mapping[field_name] = found_col

    docs = []
    for i, row in df.iterrows():
        content = f"Title: {row[title_col]}\n\nIngredients:\n{row[ingredients_col]}\n\nSteps:\n{row[steps_col]}\n"
        for opt_name, col in optional_mapping.items():
            if col:
                content += f"\n{opt_name.replace('_', ' ').capitalize()}: {row[col]}\n"
        docs.append(Document(page_content=content, metadata={"source": f"{path.name}-row-{i}"}))

    return docs

def get_all_file_paths(csv_folder, pdf_folder):
    return list(Path(csv_folder).rglob("*.csv")) + list(Path(pdf_folder).rglob("*.pdf"))

def get_modified_time(path):
    return os.path.getmtime(path)

def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return {}

def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)

def detect_and_process_new_files(csv_folder, pdf_folder, max_rows=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    indexed = load_index()
    updated_index = indexed.copy()
    new_chunks = []

    for path in get_all_file_paths(csv_folder, pdf_folder):
        path_str = str(path)
        last_modified = get_modified_time(path)

        if path_str in indexed and indexed[path_str] == last_modified:
            continue

        try:
            if path.suffix == ".csv":
                docs = load_from_csv(path, max_rows)
            elif path.suffix == ".pdf":
                docs = load_from_pdf(path)
            else:
                continue

            chunks = splitter.split_documents(docs)
            new_chunks.extend(chunks)
            updated_index[path_str] = last_modified
        except Exception as e:
            print(f" Failed to process {path.name}: {e}")

    save_index(updated_index)
    return new_chunks

def initialize_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if FAISS_INDEX_FILE.exists():
        vectorstore = FAISS.load_local(str(FAISS_INDEX_FILE), 
                                       embedding,
                                       allow_dangerous_deserialization=True)
        print(" FAISS index loaded from disk.")
    else:
        dim = len(embedding.embed_query("hello world"))
        index = faiss.IndexFlatL2(dim)
        vectorstore = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        print(" New FAISS index initialized.")

    return vectorstore

def update_vectorstore(vectorstore, new_chunks):
    if new_chunks:
        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(str(FAISS_INDEX_FILE))
        print(" FAISS index updated and saved.")
    else:
        print(" No new documents to add to FAISS index.")

def get_vectorstore():
    new_chunks = detect_and_process_new_files("data/recipes_csv", "data/recipes_pdf")
    vectorstore = initialize_vectorstore()
    update_vectorstore(vectorstore, new_chunks)
    return vectorstore


def load_llm():
    llm= ChatGroq(
        model_name="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.7,
        top_p=0.95,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ChefFriends, a smart culinary assistant. Based on the retrieved documents, provide helpful and concise answers. You should combine information if needed, rephrase in a friendly tone, and never invent recipes. If you don't know the answer, say so."),
        ("user", "{context}\n\nUser question: {query}")
    ])

    return prompt |llm | StrOutputParser()



MAX_TOKENS = 512
_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def build_context_smart(docs, query=None, max_tokens=MAX_TOKENS):
    context_parts = []
    token_count = 0

    for doc in docs:
        text = doc.page_content.strip()
        doc_tokens = _tokenizer.encode(text, truncation=False, add_special_tokens=False)
        doc_len = len(doc_tokens)

        if token_count + doc_len > max_tokens:
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 0:
                truncated_tokens = doc_tokens[:remaining_tokens]
                truncated_text = _tokenizer.decode(truncated_tokens)
                context_parts.append(truncated_text)
                token_count += remaining_tokens
            break
        else:
            context_parts.append(text)
            token_count += doc_len

    return "\n\n".join(context_parts)


