from transformers import BertTokenizer, BertForTokenClassification, BartTokenizer, BartForConditionalGeneration
import tensorflow as tf
import PyPDF2

# Load the pre-trained BERT tokenizer and model for NER
ner_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
ner_model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(ner_tokenizer.get_vocab()))

# Load the pre-trained BART tokenizer and model for summarization
summarization_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def analyze_legal_document(doc_text):
    # Perform Named Entity Recognition (NER) using the pre-trained BERT model
    tokens = ner_tokenizer.encode(doc_text, add_special_tokens=True, return_tensors='tf')
    logits = ner_model(tokens)[0]
    ner_predictions = tf.argmax(logits, axis=2)

    # Decode the tokens and their corresponding NER tags
    tokenized_text = ner_tokenizer.decode(tokens[0], skip_special_tokens=True)
    ner_tags = [ner_tokenizer.decode(tag_id) for tag_id in ner_predictions[0]]

    # Extract important data points based on NER tags (e.g., Person, Organization, Date, etc.)
    important_data = []
    current_entity = None
    current_entity_type = None
    for token, tag in zip(tokenized_text.split(), ner_tags):
        if tag.startswith('B-'):
            # Start of a new entity
            if current_entity:
                important_data.append((current_entity, current_entity_type))
            current_entity = token
            current_entity_type = tag.split('-')[1]
        elif tag.startswith('I-'):
            # Continuation of the current entity
            current_entity += ' ' + token

    if current_entity:
        important_data.append((current_entity, current_entity_type))

    # Summarize the legal document using the pre-trained BART model
    inputs = summarization_tokenizer.encode("summarize: " + doc_text, return_tensors="tf", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summarized_text = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return important_data, summarized_text

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_user_input():
    print("Please enter the path to the PDF file:")
    pdf_file_path = input()
    return pdf_file_path

# Main program
if __name__ == "__main__":
    pdf_file_path = get_user_input()
    pdf_text = read_pdf(pdf_file_path)

    important_data, summary = analyze_legal_document(pdf_text)

    print("\nImportant Data Points:")
    print(important_data)
    print("\nSummary:")
    print(summary)
