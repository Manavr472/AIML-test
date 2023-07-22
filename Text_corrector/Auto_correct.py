import string
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
from collections import Counter

class AutocorrectSuggestions:
    def __init__(self, vocab_file_path, model_checkpoint):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file_path, force_download=True)
        self.model = BertForMaskedLM.from_pretrained(model_checkpoint, force_download=True)
        self.model.eval()
        self.vocabulary = self.tokenizer.get_vocab()
        self.alphabet = set(string.ascii_lowercase)

    def _get_candidates(self, word):
        # Generate candidates by adding, deleting, or replacing one character at a time
        deletes = {word[:i] + word[i+1:] for i in range(len(word))}
        replaces = {word[:i] + c + word[i+1:] for i in range(len(word)) for c in self.alphabet}
        inserts = {word[:i] + c + word[i:] for i in range(len(word)+1) for c in self.alphabet}
        candidates = deletes | replaces | inserts
        return candidates

    def _get_probabilities(self, input_text, masked_idx):
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids)[0]
        masked_token_probs = outputs[0, masked_idx].softmax(dim=0)
        return masked_token_probs

    def _get_top_n_suggestions(self, input_text, masked_word, n=5):
        masked_idx = input_text.index(masked_word)
        probabilities = self._get_probabilities(input_text, masked_idx)
        token_ids_sorted_by_prob = torch.argsort(probabilities, descending=True)
        top_n_tokens = [self.tokenizer.decode([token_id.item()]) for token_id in token_ids_sorted_by_prob[:n]]
        return top_n_tokens

    def get_autocorrect_suggestions(self, word, n=5):
        input_text = f"The {word} is missing."
        candidates = self._get_candidates(word)
        suggestions = []

        for candidate in candidates:
            masked_text = input_text.replace(word, self.tokenizer.mask_token)
            masked_idx = masked_text.index(self.tokenizer.mask_token)
            input_text_with_candidate = masked_text.replace(self.tokenizer.mask_token, candidate)

            suggestions.append((candidate, self._get_probabilities(input_text_with_candidate, masked_idx)))

        top_suggestions = sorted(suggestions, key=lambda x: x[1].max().item(), reverse=True)[:n]
        return [suggestion[0] for suggestion in top_suggestions]

# Example usage:
vocab_file_path = 'bert-base-uncased'
model_checkpoint = 'bert-base-uncased'
spell_checker = AutocorrectSuggestions(vocab_file_path, model_checkpoint)

input_word = input("Enter your word: ")
suggestions = spell_checker.get_autocorrect_suggestions(input_word)
print(f"Autocorrect suggestions for '{input_word}':")
print(suggestions)