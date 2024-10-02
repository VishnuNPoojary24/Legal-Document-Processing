import streamlit as st
import openai
from transformers import pipeline
from groq import Groq
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os



# Define available models
model_options = {
    "Pro": "llama1",
    "Expert": "llama2",
    "Standard": "llama3",
}

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
# User selects the model
st.title("AI-Powered Contract Generator")
model_choice = st.selectbox("Select the model to generate the document:", list(model_options.keys()))

# User input for document generation
user_input = st.text_area("Enter the rules/conditions for your contract")

# def generate_with_openai(input_text):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a legal expert generating a contract."},
#             {"role": "user", "content": input_text}
#         ]
#     )
#     return response['choices'][0]['message']['content']
# def generate_with_llama(input_text):
#     prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful AI assistant. Please answer questions based on the provided documents and user query."),
#         ("user", "Documents: {documents}\nQuestions: {question}")
#     ]
#     )

#     # Initialize the LLM and output parser
#     llm = Ollama(model="llama3.1")
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser

#     # Combine the extracted document text with the user's question
#     if input_text:
#         response = chain.invoke({'documents': input_text, 'question': "Generate the document text in legal document format"})
#         return response
#     else:
#         return "Please upload documents and ask a question."

def generate_with_llama(input_text):
    # Initialize the Groq client with the API key
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Define the prompt for the model to generate the document
    user_message = f"Documents: {input_text}\nQuestions: Generate the document text in legal document format."

    # Send the request to the Llama model via Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="llama3-8b-8192",  # Specify the Llama model being used
    )

    # Extract and return the generated response
    if chat_completion and chat_completion.choices:
        return chat_completion.choices[0].message.content
    else:
        return "Please upload documents and ask a question."

def generate_with_huggingface(model_name, input_text):
    generator = pipeline("text-generation", model=model_name)
    # Using max_new_tokens instead of max_length to handle the length of generated text
    result = generator(input_text, max_new_tokens=500, truncation=True)
    return result[0]["generated_text"]

if st.button("Generate Document"):
    # Generate content using the selected model
    if model_choice == "Pro":
        document_text = generate_with_llama(user_input)
    elif model_choice=="Expert":
        document_text=generate_with_llama(user_input)
    else:
        document_text = generate_with_llama(user_input)
    
    # Convert the generated text to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, document_text)
    
    # Save the PDF to a file
    pdf_output = f"generated_contract_{model_choice.replace(' ', '_')}.pdf"
    pdf.output(pdf_output)
    
    # Provide download option
    with open(pdf_output, "rb") as file:
        st.download_button(
            label="Download Contract",
            data=file,
            file_name=pdf_output,
            mime="application/pdf"
        )
