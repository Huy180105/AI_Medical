import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.segmenter import VietnameseSegmenter

logger = get_logger(__name__)

class MedicalNERPredictor:
    def __init__(self, model_dir: str = None):
        """
        Inference class for Medical Named Entity Recognition.
        Loads the fine-tuned model and tokenizer onto the available device (GPU/CPU).
        """
        self.model_dir = model_dir or Config.OUTPUT_MODEL_DIR
        self.device = Config.DEVICE
        
        # Determine if a fine-tuned model exists, otherwise log warning and load base model
        if os.path.exists(self.model_dir) and os.path.exists(os.path.join(self.model_dir, "config.json")):
            logger.info(f"Loading fine-tuned model from {self.model_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        else:
            logger.warning(
                f"Fine-tuned model not found at {self.model_dir}. "
                f"Falling back to pretrained base model {Config.MODEL_NAME} (WARNING: Untrained weights!)."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=False)
            self.model = AutoModelForTokenClassification.from_pretrained(
                Config.MODEL_NAME,
                num_labels=Config.NUM_LABELS,
                id2label=Config.ID2LABEL,
                label2id=Config.LABEL2ID
            )
            
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model successfully loaded on device: {self.device}")

    def _align_words_to_original_text(self, original_text: str, words: list[str]) -> list[tuple[int, int]]:
        """
        Calculates original character offsets (start, end) for each segmented word token.
        """
        word_spans = []
        text_ptr = 0
        
        for word in words:
            # Clean token to match characters
            clean_word = word.replace("_", "")
            if not clean_word:
                word_spans.append((-1, -1))
                continue
                
            start_pos = -1
            match_ptr = text_ptr
            
            while match_ptr < len(original_text):
                # Check sliding window slice matching length
                slice_len = len(word)
                orig_slice = original_text[match_ptr:match_ptr + slice_len]
                orig_clean = orig_slice.replace(" ", "").replace("_", "")
                
                if clean_word.lower() in orig_clean.lower() or orig_clean.lower() in clean_word.lower():
                    # Align character by character to find exact boundary
                    temp_clean = ""
                    idx = match_ptr
                    while idx < len(original_text) and len(temp_clean) < len(clean_word):
                        char = original_text[idx]
                        if char != " " and char != "_":
                            temp_clean += char
                        idx += 1
                        
                    if temp_clean.lower() == clean_word.lower():
                        start_pos = match_ptr
                        end_pos = idx
                        
                        # Trim leading spaces and underscores from start_pos
                        while start_pos < end_pos and original_text[start_pos] in (" ", "_"):
                            start_pos += 1
                            
                        text_ptr = end_pos
                        word_spans.append((start_pos, end_pos))
                        break
                        
                match_ptr += 1
                
            if start_pos == -1:
                # Fallback: estimate position
                est_start = original_text.lower().find(clean_word.lower(), text_ptr)
                if est_start != -1:
                    start_pos = est_start
                    end_pos = est_start + len(clean_word)
                    
                    # Trim leading spaces and underscores from start_pos
                    while start_pos < end_pos and original_text[start_pos] in (" ", "_"):
                        start_pos += 1
                        
                    text_ptr = end_pos
                    word_spans.append((start_pos, end_pos))
                else:
                    word_spans.append((-1, -1))
                    
        return word_spans

    def predict(self, text: str) -> list[dict]:
        """
        Performs inference on raw text to extract clinical entities.
        """
        if not text or not text.strip():
            return []
            
        # 1. Word segmentation
        words = VietnameseSegmenter.segment_to_words(text)
        if not words:
            return []
            
        # 2. Tokenize segmented words
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=Config.MAX_LEN,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 3. Model forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=2)
            
        # Convert tensors to numpy lists
        logits = logits.cpu().squeeze(0).numpy()
        probabilities = probabilities.cpu().squeeze(0).numpy()
        
        # Count subwords per word for alignment without word_ids()
        subword_counts = []
        for word in words:
            subword_counts.append(len(self.tokenizer.tokenize(word)))
            
        # 4. Map subword predictions back to segmented words
        # Only use prediction of the first subword of each word
        word_preds = []
        word_probs = []
        
        current_idx = 1  # Skip BOS token
        for word, count in zip(words, subword_counts):
            if current_idx < len(logits):
                pred_id = logits[current_idx].argmax()
                pred_label = Config.ID2LABEL[pred_id]
                prob = probabilities[current_idx][pred_id]
                word_preds.append(pred_label)
                word_probs.append(float(prob))
            else:
                word_preds.append("O")
                word_probs.append(0.0)
            current_idx += count
            
        # Guarantee word_preds is aligned with words list
        if len(word_preds) < len(words):
            words = words[:len(word_preds)]
        elif len(word_preds) > len(words):
            word_preds = word_preds[:len(words)]
            word_probs = word_probs[:len(words)]
            
        # Calculate character spans in original text
        word_spans = self._align_words_to_original_text(text, words)
        
        # 5. Reconstruct entities from BIO sequence
        entities = []
        current_entity = None
        
        for i, (word, pred_tag, prob, (w_start, w_end)) in enumerate(zip(words, word_preds, word_probs, word_spans)):
            if pred_tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                ent_type = pred_tag.split("-")[1]
                current_entity = {
                    "text": text[w_start:w_end] if w_start != -1 else word.replace("_", " "),
                    "type": ent_type,
                    "score": prob,
                    "start": w_start,
                    "end": w_end,
                    "word_indices": [i]
                }
            elif pred_tag.startswith("I-") and current_entity:
                ent_type = pred_tag.split("-")[1]
                # Verify tag type matches current entity type
                if ent_type == current_entity["type"]:
                    current_entity["word_indices"].append(i)
                    if w_end != -1:
                        # Extend original character end pointer
                        current_entity["end"] = w_end
                        current_entity["text"] = text[current_entity["start"]:w_end]
                    else:
                        current_entity["text"] += " " + word.replace("_", " ")
                    # Average confidence score
                    current_entity["score"] = (current_entity["score"] + prob) / 2.0
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    
        if current_entity:
            entities.append(current_entity)
            
        return entities
