
import os
import io
import streamlit as st
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

st.set_page_config(
    page_title="CampusCompass AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CampusCompass AI")
st.caption("Semantic RAG assistant for campus documents")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    db = chromadb.PersistentClient(path="./chroma_db")
    return db.get_or_create_collection(
        name="campus_documents"
    )

model = load_embedding_model()
collection = load_vector_db()

uploaded_files = st.file_uploader(
    "Upload college documents (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Index Documents"):
    with st.spinner("Reading documents and creating embeddings..."):
        indexed_chunks = 0

        for uploaded_file in uploaded_files:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                # Split each page into smaller chunks
                words = text.split()
                chunks = [
                    " ".join(words[i:i + 180])
                    for i in range(0, len(words), 150)
                ]

                for chunk_num, chunk in enumerate(chunks):
                    chunk_id = (
                        f"{uploaded_file.name}_p{page_num}_c{chunk_num}"
                    )
                    embedding = model.encode(chunk).tolist()

                    collection.upsert(
                        ids=[chunk_id],
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{
                            "document": uploaded_file.name,
                            "page": page_num
                        }]
                    )
                    indexed_chunks += 1

        st.session_state["indexed"] = True
        st.success(f"Indexed {indexed_chunks} document chunks.")

question = st.text_input(
    "Ask a question",
    placeholder="What are the placement eligibility criteria?"
)

if st.button("Find Answer", type="primary") and question.strip():
    if collection.count() == 0:
        st.warning("Upload and index a PDF first.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            query_embedding = model.encode(question).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(4, collection.count())
            )

            docs = results["documents"][0]
            metadata = results["metadatas"][0]

            context = "\n\n".join(
                f"Source: {m['document']}, page {m['page']}\n{d}"
                for d, m in zip(docs, metadata)
            )

            prompt = f"""
You are CampusCompass AI.

Answer only from the retrieved excerpts below.
Do not invent campus rules or eligibility criteria.
If the excerpts do not contain enough evidence, say so clearly.
Cite document names and page numbers in your answer.

EXCERPTS:
{context}

QUESTION:
{question}
"""

            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                st.subheader("Answer")
                st.markdown(response.text)

                st.subheader("Retrieved sources")
                for d, m in zip(docs, metadata):
                    with st.expander(
                        f"{m['document']} — Page {m['page']}"
                    ):
                        st.write(d)

            except Exception as error:
                st.error(f"Answer generation failed: {error}")

if collection.count() == 0:
    st.info("Upload a college PDF and click Index Documents to begin.")

# ---------- Placement Eligibility Checker ----------

st.divider()
st.header("🎯 Placement Eligibility Checker")

st.caption("Based on our fictional demo college handbook.")

with st.form("eligibility_form"):
    cgpa = st.number_input(
        "Your CGPA (out of 10)",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

    backlogs = st.number_input(
        "Number of active backlogs",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    year = st.selectbox(
        "Graduation year",
        [2026, 2027, 2028, 2029, 2030],
        index=1
    )

    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    checks = [
        ("Minimum CGPA of 7.5", cgpa >= 7.5),
        ("Zero active backlogs", backlogs == 0),
        ("Graduation year 2027 or 2028", year in [2027, 2028])
    ]

    for rule, passed in checks:
        if passed:
            st.success(f"✅ {rule}: Passed")
        else:
            st.error(f"❌ {rule}: Not met")

    if all(passed for _, passed in checks):
        st.success("You meet all criteria in the demo policy.")
    else:
        st.warning("You do not meet all demo criteria.")

    st.caption(
        "Demo only. These are fictional rules, "
        "not official college or employer requirements."
    )


# ---------- Cross-Document Comparison ----------

st.divider()
st.header("📑 Compare College Documents")

st.write(
    "Upload two placement notices to compare their eligibility rules."
)

col1, col2 = st.columns(2)

with col1:
    old_doc = st.file_uploader(
        "Upload older notice (PDF)",
        type=["pdf"],
        key="old_notice"
    )

with col2:
    new_doc = st.file_uploader(
        "Upload newer notice (PDF)",
        type=["pdf"],
        key="new_notice"
    )

if st.button("Compare Documents"):
    if not old_doc or not new_doc:
        st.warning("Please upload both PDF documents.")
    else:
        from pypdf import PdfReader
        import io

        def extract_pdf_text(file):
            reader = PdfReader(io.BytesIO(file.getvalue()))
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        old_text = extract_pdf_text(old_doc)
        new_text = extract_pdf_text(new_doc)

        if not old_text.strip() or not new_text.strip():
            st.error("Could not extract text. Please check both PDFs.")
        else:
            with st.spinner("Comparing document rules with Gemini..."):
                try:
                    comparison_prompt = f"""
Compare these two college placement notices.
Identify changes in CGPA, backlogs, graduation year,
eligible branches, deadlines, and required documents.
Only report differences supported by the text.
If a detail is missing, say it is not specified.
Do not invent policies.

OLDER NOTICE:
{old_text[:12000]}

NEWER NOTICE:
{new_text[:12000]}
"""
                    result = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=comparison_prompt
                    )

                    st.subheader("Comparison Results")
                    st.markdown(result.text)

                except Exception as error:
                    st.error(f"Comparison failed: {error}")
