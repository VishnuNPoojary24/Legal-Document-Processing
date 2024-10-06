# Legal Document Processing System

The Legal Document Processing System is a tool designed to handle and process legal documents, allowing users to upload, query, and generate documents based on input. This system leverages advanced NLP models and provides an intuitive interface for users to interact with their legal data.

## Project Overview

The **Legal Document Processing System** consists of two main components:
1. **Homepage**: Users can upload legal documents, query their contents, and receive model-generated responses.
2. **Document Generator**: Based on user input, the system can generate new legal documents and export them in PDF format.

## Features

### Homepage
- **File Upload**: Supports uploading PDFs, DOCX files, and image files (JPG, PNG).
- **Query Input**: Allows users to input queries related to the uploaded documents.
- **Model Selection**: Supports multiple NLP models for processing queries.
- **Response Display**: Displays responses from the selected model.
- **Query History**: Shows the history of past queries and responses with timestamps.

### Document Generator
- **Model Selection**: Users can choose from various NLP models for document generation.
- **Text Input**: Allows users to enter rules or conditions for generating legal documents.
- **Document Creation**: Generates documents based on user inputs and provides a downloadable PDF.

## Architecture

The system utilizes a Python-based backend with key libraries for text extraction and generation, and a Streamlit interface for the frontend. Query histories are stored in an SQLite database.

**Backend**:
- **Text Extraction**: Handles PDFs, DOCX, and image files using `PyPDF2`, `docx`, and `EasyOCR`.
- **Text Generation**: Utilizes models like OpenAI’s `gpt-3.5-turbo` and `Llama 3.1`.
- **Database**: SQLite for storing queries, responses, and timestamps.

https://legal-document-processing.streamlit.app/
