import os
import re
import openai
import nltk
import streamlit as st
import sqlite3
from datetime import datetime
from groq import Groq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document
import PyPDF2
import easyocr
from transformers import pipeline  # New import for DistilBERT
#from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Run the script to download the necessary NLTK resources
os.system('python download_nltk_resources.py')

# Set up Streamlit page configuration
st.set_page_config(page_title="Legal Document Processor", page_icon=":page_facing_up:")

# Initialize NLP tools

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

def tokenize_text(text):
    """Tokenize text into words and sentences."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

def remove_stopwords(words):
    """Remove stop words from the tokenized text."""
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def lemmatize_words(words):
    """Lemmatize words to their base form."""
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def preprocess_text(text):
    """Combine all preprocessing steps."""
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text)
    words, sentences = tokenize_text(normalized_text)
    words = remove_stopwords(words)
    words = lemmatize_words(words)  # Using lemmatization instead of stemming
    preprocessed_text = ' '.join(words)
    return preprocessed_text, text  # Return preprocessed and raw text

#def query_chatgpt(prompt):
    """Query OpenAI's ChatGPT API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()
#def llama_langchain(input_text,document_text):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Please answer questions based on the provided documents and user query."),
        ("user", "Documents: {documents}\nQuestions: {question}")
    ]
    )
     # Initialize the LLM and output parser
    llm = Ollama(model="llama3.1")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    if input_text and document_text:
        response = chain.invoke({'documents': document_text, 'question': input_text})
        return response
    else:
        return "Please upload documents and ask a question."
    
#def create_database():
    """Create the database and the queries table if it doesn't exist."""
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


def llama_langchain(input_text, document_text):
    # Initialize the Groq client with the API key
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    # Combine document text and input text into a formatted question
    user_message = f"Documents: {document_text}\nQuestions: {input_text}"

    # Send the request to the Llama model via Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
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
    #create_database()
    
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
            margin-top: 10px; /* Move the upload section up by 20px (adjust as needed) */
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
    model_options = ["AI-Pro ", "AI-Expert", "Standard"]
    selected_model = st.selectbox("", model_options)


    st.markdown(
    """
    <style>
    .upload-label {
        color: white;
        margin-bottom: -10px;  /* Adjust this value to control the gap */
    }
    .stFileUploader {
        margin-top: 0px;  /* Remove any additional margin at the top of the file uploader */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    # File uploader section with custom color and adjusted position
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<p class="upload-label">Upload files (PDF, DOCX, Image)</p>', unsafe_allow_html=True)
    files = st.file_uploader("", type=['pdf', 'docx', 'jpg', 'jpeg', 'png'], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Query input section with custom color and reduced spacing
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<p class="query-label">Enter your query:</p>', unsafe_allow_html=True)
    input_query = st.text_input("", key="query_input", placeholder="Type your query here...", help="")
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button('Submit', key='submit_button'):
        st.write(f'You entered: {input_query}')

    if files:
        # Create the uploads directory if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        all_texts_cleaned = []
        all_texts_raw = []
        for file in files:
            file_path = os.path.join("uploads", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            try:
                extracted_text = extract_text(file_path)
                cleaned_text, raw_text = preprocess_text(extracted_text)
                all_texts_cleaned.append(cleaned_text)
                all_texts_raw.append(raw_text)
            except ValueError as e:
                st.error(f"Error processing {file.name}: {e}")
                continue

        if all_texts_cleaned and all_texts_raw:
            combined_text_cleaned = ' '.join(all_texts_cleaned)  # For OpenAI models
            combined_text_raw = ' '.join(all_texts_raw)          # For DistilBERT

            if input_query:
                if selected_model.startswith("AI"):
                    response = llama_langchain(input_text=input_query, document_text=combined_text_raw)
                    response_text = response
                elif selected_model == "Standard":
                    response = llama_langchain(input_text=input_query, document_text=combined_text_raw)
                    response_text = response
                else:
                    response_text = "Selected model is not supported."

                # Display the response with custom color
                st.markdown('<p class="response-label">Response:</p>', unsafe_allow_html=True)
                st.write(response_text)

                # Store the model, query, response, and timestamp in the database
                insert_query_response(selected_model, input_query, response_text)

    # Sidebar for displaying query history
    st.sidebar.header("Query History")
    history = fetch_query_history()

    if history:
        with st.sidebar:
            st.markdown("<div class='query-history'>", unsafe_allow_html=True)
            display_history = [f"[{row[1]}] {row[2]} (Timestamp: {row[3]})" for row in history]
            selected_display = st.selectbox("Select a query to view response", display_history)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Fetch the query ID based on the selected display string
            selected_query_id = None
            for row in history:
                display_str = f"[{row[1]}] {row[2]} (Timestamp: {row[3]})"
                if display_str == selected_display:
                    selected_query_id = row[0]
                    break

            if selected_query_id:
                response = fetch_response_for_query(selected_query_id)
                if response:
                    st.markdown("<div class='response'>", unsafe_allow_html=True)
                    st.write(f"**Response:** {response}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    


if __name__ == "__main__":
    main()
