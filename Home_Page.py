import os
import re
import openai
import streamlit as st
import sqlite3
from datetime import datetime
from groq import Groq
import spacy
from docx import Document
import PyPDF2
import easyocr
from transformers import pipeline  # New import for DistilBERT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Set up Streamlit page configuration
st.set_page_config(page_title="Legal Document Processor", page_icon=":page_facing_up:")

# Initialize EasyOCR reader object
reader = easyocr.Reader(['en'])

# Initialize DistilBERT QA pipeline with caching
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_pipeline()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def extract_text_from_image(image_path):
    """Extract text from an image file using EasyOCR."""
    results = reader.readtext(image_path)
    text = "\n".join([result[1] for result in results])
    return text

def extract_text_from_docx(docx_path):
    """Extract text from a .docx file."""
    doc = Document(docx_path)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text

def extract_text(file_path):
    """Extract text from a file based on its extension."""
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        return extract_text_from_image(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type: {}".format(extension))

def clean_text(text):
    """Clean the text by removing unwanted characters and whitespace."""
    text = text.strip()  # Remove leading and trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def normalize_text(text):
    """Normalize the text to a standard format."""
    text = text.lower()  # Convert to lowercase
    return text

def preprocess_text(text):
    """Combine all preprocessing steps using spaCy."""
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    
    # Process the text with spaCy
    doc = nlp(normalized_text)
    
    # Remove stop words and lemmatize
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop]
    
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text, text  # Return preprocessed and raw text

def llama_langchain(input_text, document_text):
    """Process the input text with Llama model via Groq."""
    # Initialize the Groq client with the API key
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    # Combine document text and input text into a formatted question
    user_message = f"Documents: {document_text}\nQuestions: {input_text}"

    # Send the request to the Llama model via Groq
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": user_message,
        }],
        model="llama3-8b-8192",  # Use the desired Llama model
    )
    
    # Return the generated response
    return chat_completion.choices[0].message.content if chat_completion else "No response received."

def insert_query_response(model, query, response):
    """Insert the model, query, its response, and the current timestamp into the database."""
    conn = sqlite3.connect('queries.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO queries (model, query, response, timestamp) VALUES (?, ?, ?, ?)",
              (model, query, response, timestamp))
    conn.commit()
    conn.close()

def fetch_query_history():
    """Fetch query history from the database."""
    conn = sqlite3.connect('queries.db')
    c = conn.cursor()
    c.execute("SELECT id, model, query, timestamp FROM queries ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()
    return history

def fetch_response_for_query(query_id):
    """Fetch response for a specific query from the database."""
    conn = sqlite3.connect('queries.db')
    c = conn.cursor()
    c.execute("SELECT response FROM queries WHERE id = ?", (query_id,))
    response = c.fetchone()
    conn.close()
    return response[0] if response else None

def main():
    # Create the database if it doesn't exist
    if not os.path.exists("queries.db"):
        conn = sqlite3.connect('queries.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS queries
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      model TEXT NOT NULL,
                      query TEXT NOT NULL,
                      response TEXT NOT NULL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    # Apply custom CSS for dark mode
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Custom styling
    st.markdown(
        """
        <style>
        .title {
            color: red;
        }
        .upload-label, .query-label {
            color: lightgrey;
        }
        .response-label {
            color: white;
        }
        .query-container {
            display: flex;
            flex-direction: column;
            gap: 0px; /* Remove space between label and input box */
        }
        .query-label {
            margin-bottom: -20px; /* Remove space below the label */
        }
        .query-input {
            margin-top: 20px; /* Remove space above the input box */
        }
        .upload-container {
            margin-top: 10px; /* Move the upload section up by 10px (adjust as needed) */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Title with custom color
    st.markdown('<h1 class="title">Legal Document Processor by ENTERTAINMENT TECHNOLOGISTS</h1>', unsafe_allow_html=True)

    # Model selection
    st.markdown(
        """
        <style>
        .model-select-label {
            color: white;
            margin-bottom: -30px;
        }
        .stSelectbox > div {
            margin-top: 0px;  /* Remove any additional margin at the top of the selectbox */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the selectbox with the custom label
    st.markdown('<p class="model-select-label">Select the model to use for answering:</p>', unsafe_allow_html=True)
    model_options = ["AI-Pro", "AI-Expert", "Standard"]
    selected_model = st.selectbox("", model_options)

    # File uploader section with custom color and adjusted position
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<p class="upload-label">Upload files (PDF, DOCX, Image)</p>', unsafe_allow_html=True)
    files = st.file_uploader("", type=['pdf', 'docx', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if files:
        document_text = ""
        for uploaded_file in files:
            # Read and extract text from the uploaded files
            try:
                text = extract_text(uploaded_file)
                document_text += text + "\n\n"  # Combine text from all uploaded files
            except ValueError as e:
                st.error(str(e))

        st.text_area("Extracted Document Text", value=document_text, height=300)

        # User input for querying
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        st.markdown('<p class="query-label">Enter your query:</p>', unsafe_allow_html=True)
        user_query = st.text_input("", placeholder="Type your question here...", key="user_query", className="query-input")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Submit Query"):
            if user_query:
                preprocessed_text, raw_text = preprocess_text(document_text)
                response = llama_langchain(user_query, raw_text)  # Use raw text for response
                insert_query_response(selected_model, user_query, response)  # Store query and response

                # Display the response
                st.markdown('<p class="response-label">Response:</p>', unsafe_allow_html=True)
                st.text_area("", value=response, height=200)

                # Display chat history
                st.markdown('<h3>Query History</h3>', unsafe_allow_html=True)
                history = fetch_query_history()
                for query in history:
                    query_id, model, query_text, timestamp = query
                    query_display = f"ID: {query_id}, Model: {model}, Query: {query_text}, Timestamp: {timestamp}"
                    if st.button(f"Show Response for Query ID {query_id}"):
                        response_text = fetch_response_for_query(query_id)
                        st.text_area("", value=response_text, height=200)
                    st.write(query_display)

if __name__ == "__main__":
    main()
