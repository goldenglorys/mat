import streamlit as st
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
from cryptography.fernet import Fernet
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key management: Load or generate encryption key
KEY_FILE = "encryption_key.key"


def load_or_generate_key():
    """Load existing key or generate a new one if it doesn't exist"""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key


ENCRYPTION_KEY = load_or_generate_key()
cipher = Fernet(ENCRYPTION_KEY)


# =============================================
# Part 1: Chunking & Vectorization
# =============================================
def load_data(file_path):
    """Load and preprocess product data"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        for product in data:
            try:
                product["co2_emissions"] = float(product.get("co2_emissions", 0))
                recycled = product.get("recycled_materials", "0%").rstrip("%")
                product["recycled_materials"] = float(recycled) if recycled else 0.0
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Error processing product {product.get('product_name', 'Unknown')}: {e}"
                )
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []


def format_materials(materials_dict):
    """Format materials dictionary to string"""
    if not materials_dict:
        return "None"
    return ", ".join([f"{k} ({v})" for k, v in materials_dict.items()])


def create_product_chunk(product):
    """Convert a single product to a text chunk for embedding"""
    materials = format_materials(product.get("materials_used", {}))
    return (
        f"Product: {product.get('product_name', 'Unnamed')}. "
        f"Category: {product.get('category', 'Unknown')}. "
        f"Description: {product.get('short_description', 'No description')}. "
        f"Materials: {materials}. "
        f"Thickness: {product.get('thickness', 'Unknown')}. "
        f"CO2 Emissions: {product.get('co2_emissions', 0):.2f} kg. "
        f"Recycled Materials: {product.get('recycled_materials', 0)}%."
    )


def create_product_chunks(products):
    """Convert products to text chunks for embedding"""
    return [create_product_chunk(product) for product in products]


@st.cache_resource
def load_embedding_model():
    """Load and cache sentence transformer model"""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_function():
    """Get embedding function for ChromaDB"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =============================================
# Part 2: RAG Stuff
# =============================================
class VectorDB:
    def __init__(self, products):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedder = load_embedding_model()
        self.products = products
        self.collection = self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or verify collection"""
        collection = self.client.get_or_create_collection(
            name="products",
            embedding_function=get_embedding_function(),
        )

        # Check if collection is empty or needs reinitialization
        existing_ids = collection.get()["ids"]
        if not existing_ids:
            self._populate_collection(collection)
        else:
            # Verify decryption works; if not, reinitialize
            try:
                sample_doc = collection.get(ids=[existing_ids[0]])["documents"][0]
                cipher.decrypt(sample_doc.encode()).decode()
            except Exception as e:
                logger.warning(
                    f"Decryption failed for existing data: {e}. Reinitializing collection."
                )
                self.client.delete_collection("products")
                collection = self.client.create_collection(
                    name="products",
                    embedding_function=get_embedding_function(),
                )
                self._populate_collection(collection)

        return collection

    def _populate_collection(self, collection):
        """Populate collection with embeddings"""
        chunks = create_product_chunks(self.products)
        embeddings = self.embedder.encode(chunks, show_progress_bar=True).tolist()
        encrypted_chunks = [cipher.encrypt(chunk.encode()).decode() for chunk in chunks]

        collection.add(
            documents=encrypted_chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(self.products))],
            metadatas=[
                {"category": p.get("category", "Unknown")} for p in self.products
            ],
        )

    def search(self, query, k=5):
        """Search for similar products"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        # Decrypt documents before returning
        if results and "documents" in results and results["documents"]:
            try:
                decrypted_docs = [
                    cipher.decrypt(doc.encode()).decode()
                    for doc in results["documents"][0]
                ]
                results["documents"] = [decrypted_docs]
            except Exception as e:
                logger.error(f"Decryption failed during search: {e}")
                results["documents"] = [[]]

        return results


@st.cache_resource
def load_generator():
    """Load and cache text generation model"""
    return pipeline("text2text-generation", model="google/flan-t5-base")


# =============================================
# Part 3: Ethical Considerations
# =============================================
def extract_recycled_percentage(doc_text):
    """Extract recycled materials percentage from document text"""
    try:
        return float(doc_text.split("Recycled Materials: ")[1].split("%")[0])
    except (IndexError, ValueError):
        return 0.0


def extract_co2_emissions(doc_text):
    """Extract CO2 emissions from document text"""
    try:
        return float(doc_text.split("CO2 Emissions: ")[1].split(" kg")[0])
    except (IndexError, ValueError):
        return 0.0


def apply_ethical_filters(results, min_recycled=30):
    """Filter results based on ethical thresholds"""
    filtered = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        recycled_value = extract_recycled_percentage(doc)
        if recycled_value >= min_recycled:
            filtered.append(
                {
                    "document": doc,
                    "recycled_percent": recycled_value,
                    "metadata": metadata,
                }
            )

    return sorted(filtered, key=lambda x: x["recycled_percent"], reverse=True)


def audit_dataset(products):
    """Basic audit for dataset biases"""
    categories = [p.get("category", "Unknown") for p in products]
    recycled_values = [p.get("recycled_materials", 0) for p in products]

    return {
        "category_counts": pd.Series(categories).value_counts().to_dict(),
        "recycled_mean": (
            sum(recycled_values) / len(recycled_values) if recycled_values else 0
        ),
    }


@st.dialog("Security & Ethical Notes")
def show_security_ethics():
    st.write(
        """
        ### Security Risks in Using a Vector Database for Product Profiles
        1. **Unauthorized Access**: If the vector database (e.g., ChromaDB) is stored locally or on an unsecured server, unauthorized users could access sensitive product data.
        - **Mitigation**: Encrypting data at rest (as implemented with Fernet) and also in transit, and restricting file system permissions. For deployment, using authentication and hosting on a secure server with access controls like (RBAC) is decent.
        2. **Data Leakage**: The embeddings might inadvertently encode and expose sensitive product information, also public deployment without proper controls could expose proprietary product profiles as well.
        - **Mitigation**: Using API gateways with rate limiting and auth tokens. Masking sensitive data fields and avoiding the exposure of raw data contents by returning only processed results.
        3. **Query Injection and Prompt Engineering Attacks**: Malicious queries could be designed to manipulate vector search results or extract unintended information
        - **Mitigation**: Input sanitization to prevent potential query injection or vector db manipulation, and using templates for query generation to prevent prompt manipulation.

        ### Handling Potential Biases in the Dataset
        In this simple implementation, I've implemented two mechanism across two steps in the pipeline, which are: 
        1. **Preprocessing**: Auditing the dataset for imbalances (e.g., over-representation of certain categories or low recycled material products). 
        - **Current Approach**: The 'Dataset Audit' section shows category distribution and average recycled materials to identify biases.
        2. **Filtering**: Applying user-defined ethical filters (e.g., minimum recycled material percentage) to prioritize sustainable products and mitigate bias toward less eco-friendly options.
        - **Current Approach**: The slider filters results post-retrieval to enforce sustainability thresholds.

        **Algorithmic Fairness in Production Systems**: For a production system, implementing algorithmic fairness would better improve bias mitigation. 
        - Implementing fairness metrics to evaluate retrieval results across product categories
        - Using in-depth explainable AI techniques to provide transparency in why certain products were retrieved
        - Applying bias detection algorithms to monitor and report potential biases in search results
        - Periodically auditing the system with diverse query sets to identify and address systemic biases
        """
    )


def format_materials_for_display(materials_dict):
    """Format materials dictionary for display in dataframe"""
    if not materials_dict:
        return "None"
    return ", ".join([f"{k}: {v}" for k, v in materials_dict.items()])


def display_product_details(result):
    """Display product details in expandable section"""
    with st.expander(result["document"].split(". ")[0]):
        cols = st.columns(3)
        cols[0].write(f"**Recycled Materials:** {result['recycled_percent']}%")
        co2 = extract_co2_emissions(result["document"])
        cols[1].write(f"**CO2 Emissions:** {co2:.2f} kg")
        cols[2].write(f"**Details:** {result['document']}")


def process_search_results(raw_results, min_recycled, generator):
    """Process and display search results"""
    filtered_results = apply_ethical_filters(raw_results, min_recycled)

    if filtered_results:
        st.subheader(f"Found {len(filtered_results)} relevant products:")
        combined_docs = " ".join([r["document"] for r in filtered_results])
        rag_prompt = f"Provide a concise summary of these products for sustainability: {combined_docs}"
        rag_summary = generator(rag_prompt, max_length=150)[0]["generated_text"]
        st.write(f"**AI Summary of Results**: {rag_summary}")

        for result in filtered_results:
            display_product_details(result)
    else:
        st.warning("No products match your criteria.")

    st.info(":lock: Data is stored locally and encrypted at rest.")


# =============================================
# Streamlit UI Implementation
# =============================================
def display_sidebar():
    """Display sidebar information"""
    with st.sidebar:
        st.header("About This System")
        st.write(
            """
        ### Model Choices
        - **Embedding**: `all-MiniLM-L6-v2` is lightweight (22M parameters), fast and designed for semantic similarity tasks.
        - **Generation**: `flan-t5-base` (248M parameters) balances performance and efficiency for summarization tasks.

        ### Security Notes
        Data is stored locally at `./chroma_db` and encrypted using Fernet symmetric encryption. The key is stored in 'encryption_key.key' and reused across sessions.
        On production system, it's best to use environment variables or a key management service and regularly rotate keys and audit access.
        """
        )
        if st.button("Security & Ethical Notes"):
            show_security_ethics()


def display_product_data(products):
    """Display product data in expandable section"""
    with st.expander("View All Products"):
        display_df = pd.DataFrame(products)
        display_df["materials_used"] = display_df["materials_used"].apply(
            format_materials_for_display
        )
        st.dataframe(display_df.drop(columns=["materials_used"], errors="ignore"))


def display_audit_data(audit):
    """Display audit data in expandable section"""
    with st.expander("Dataset Audit"):
        st.write(f"Category Distribution: {audit['category_counts']}")
        st.write(f"Average Recycled Materials: {audit['recycled_mean']:.2f}%")


def main():
    st.title("Eco Construction Product Search")
    display_sidebar()

    products = load_data("dataset.json")
    if not products:
        st.error("Failed to load dataset. Please check the file.")
        return

    vector_db = VectorDB(products)
    generator = load_generator()

    display_product_data(products)

    audit = audit_dataset(products)
    display_audit_data(audit)

    sample_queries = [
        "Show me waterproof flooring options",
        "Find thick bamboo flooring under 15mm",
        "Flooring with aluminum oxide coating",
        "Insulation made from recycled bottles",
        "Thin insulation",
        "Plant-based insulation materials",
        "Paint with zero VOCs",
        "Clay-based interior wall coatings",
        "Self-cleaning exterior paint",
        "Products with over 80% recycled content",
        "Materials with negative carbon footprint",
        "Flooring using agricultural waste",
        "Vinyl flooring with calcium carbonate under 1.5kg CO2",
        "Natural fiber wall panels under 10kg CO2 emissions",
        "Products containing both bamboo and recycled materials",
    ]

    query_option = st.selectbox(
        "Select a query or choose 'Custom' to enter your own:",
        ["Custom"] + sample_queries,
        index=0,
    )

    if query_option == "Custom":
        query = st.text_input(
            "Enter your custom query:",
            key="custom_query_input",
        )
    else:
        query = st.text_input(
            "Edit the query if needed:",
            value=query_option,
            key="edited_query_input",
        )

    min_recycled = st.slider("Minimum recycled material (%)", 0, 100, 30)
    search_button = st.button("Search")

    if search_button and query:
        raw_results = vector_db.search(query)
        process_search_results(raw_results, min_recycled, generator)


if __name__ == "__main__":
    main()
