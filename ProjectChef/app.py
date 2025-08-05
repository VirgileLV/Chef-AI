import streamlit as st
from data_pipeline import (
    get_vectorstore,
    load_llm,
    build_context_smart
)

# Generate answer based on vectorstore retrieval and LLM
def generate_answer(chain, query, vectorstore):
    docs = vectorstore.similarity_search(query, k=4)
    context = build_context_smart(docs, query=query)
    
    response = chain.invoke({
        "context": context,
        "query": query
    })
    
    return response, docs

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="ChefFriends", layout="wide")
st.title("👨‍🍳 ChefFriends — Your AI Cooking Assistant")
st.write("Upload your cooking documents (CSV or PDF) and ask anything about recipes!")

# Sidebar File Upload
st.sidebar.header("📁 Upload Recipe Documents")
uploaded_files = st.sidebar.file_uploader("Upload CSV or PDF", type=["csv", "pdf"], accept_multiple_files=True)

# Save uploaded files to expected folders
if uploaded_files:
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            with open(f"data/recipes_csv/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        elif file.name.endswith(".pdf"):
            with open(f"data/recipes_pdf/{file.name}", "wb") as f:
                f.write(file.getbuffer())
    st.sidebar.success("Files saved. Please reload the app to reprocess the data.")

# Load LLM and vectorstore
with st.spinner("Loading model and vector store..."):
    chain = load_llm()
    vectorstore = get_vectorstore()

# User query input
st.header("💬 Ask your cooking question")
user_query = st.text_input("Type your question below and press Enter", placeholder="e.g., What's a healthy dessert under 20 minutes?")

if user_query:
    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(chain, user_query, vectorstore)
        st.subheader("🍽️ Answer")
        st.write(answer)

        st.subheader("📚 Source Documents")
        for i, doc in enumerate(sources):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source')}")
            st.code(doc.page_content[:1000])
