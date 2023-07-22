import PyPDF2
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from summa.summarizer import summarize

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # 6 classes for document types

# Define document types
document_types = {
    0: "Contract",
    1: "Agreement",
    2: "Deed",
    3: "Law",
    4: "Regulation",
    5: "Court Judgment"
}

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
    return text

def classify_document_type(text):
    # Tokenize and process text for BERT input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Perform document classification using the pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

def generate_summary(document_text, ratio=0.2):
    # Generate the summary using Gensim's TextRank algorithm
    summary = summarize(document_text, ratio=ratio)

    return summary

def analyze_legal_document(doc_text, doc_type):
    # Add code to extract important data points and clauses based on the document type.
    # You can implement NER, rule-based systems, or other techniques for information extraction.

    print("Document Type:", document_types[doc_type])
    print("Summary:")
    summary = generate_summary(pdf_text)
    print(summary)

# Example usage:
pdf_file_path = 'C:\\Users\\prana\\Downloads\\232755097.pdf'
pdf_text = read_pdf(pdf_file_path)

# Classify the document type
doc_type = classify_document_type(pdf_text)

# Analyze the legal document based on its type
analyze_legal_document(pdf_text, doc_type)