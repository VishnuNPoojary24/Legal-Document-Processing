import os
import re
import openai
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document
import PyPDF2
import easyocr
from PIL import Image

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize EasyOCR reader object
reader = easyocr.Reader(['en'])  # You can specify other languages if needed

# OpenAI API key
openai.api_key = 'add API-Key'

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
    text = clean_text(text)
    text = normalize_text(text)
    words, sentences = tokenize_text(text)
    words = remove_stopwords(words)
    words = lemmatize_words(words)  # Using lemmatization instead of stemming
    return ' '.join(words), sentences  # Return both cleaned words and sentences

def query_chatgpt(prompt):
    """Query OpenAI's ChatGPT API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def main():
    st.title("Legal Document Processor by ENTERTAINMENT TECHNOLOGISTS")

    file = st.file_uploader("Upload a file (PDF, DOCX, Image)", type=['pdf', 'docx', 'jpg', 'jpeg', 'png'])
    
    if file is not None:
        # Create the uploads directory if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            extracted_text = extract_text(file_path)
            cleaned_text, _ = preprocess_text(extracted_text)
            st.write("Preprocessed Text:")
            st.write(cleaned_text)
            
            input_query = st.text_input("Enter your query:")
            if input_query:
                query = cleaned_text + "\n" + input_query
                response_text = query_chatgpt(query)
                st.write("Response:")
                st.write(response_text)
        
        except ValueError as e:
            st.error(e)

if __name__ == "__main__":
    main()
