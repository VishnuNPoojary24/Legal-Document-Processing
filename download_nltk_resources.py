import nltk

def download_nltk_resources():
    # Download 'punkt' tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    # Download 'stopwords' corpus
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

if __name__ == "__main__":
    download_nltk_resources()
