import PyPDF2
from transformers import BertTokenizer, BertForSequenceClassification, BertLMHeadModel
import torch
import nltk
nltk.download('punkt')

# Load the pre-trained BART tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', max_length=512)
model = BertLMHeadModel.from_pretrained('bert-large-uncased', num_labels=6)  # 6 classes for document types


# Define document types
document_types = {
    0: "Contract",
    1: "Agreement",
    2: "Deed",
    3: "Law",
    4: "Regulation",
    5: "Court Judgment",
    1012: "Unknown"
}

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def classify_document_type(text):
    # Tokenize and process text for BERT input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Perform document classification using the pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits for the [CLS] token, which represents the classification result for the entire sequence
    cls_logits = outputs.logits[:, 0, :]

    # Get the predicted class by finding the class index with the highest probability
    predicted_class = torch.argmax(cls_logits, dim=1).item()

    return predicted_class

def summarize_document(text):
    # Tokenize the input text for BART input
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Perform text summarization using the pre-trained BART model
    with torch.no_grad():
        summary_ids = model.generate(inputs, max_length=1500, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary IDs to get the summarized text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary_text

def analyze_legal_document(doc_text, doc_type):
    print("Document Type:", document_types[doc_type])
    print("Summary:")
    summarized_text = summarize_document(doc_text)
    print(summarized_text)

# Example usage:
pdf_file_path = 'C:\\Users\\prana\\Downloads\\232755097.pdf'
pdf_text = read_pdf(pdf_file_path)

# Classify the document type
doc_type = classify_document_type(pdf_text)

# Analyze the legal document based on its type
analyze_legal_document(pdf_text, doc_type)