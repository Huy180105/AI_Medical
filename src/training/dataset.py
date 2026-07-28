import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MedicalNERDataset(Dataset):
    def __init__(self, json_file_path: str, tokenizer: AutoTokenizer, max_len: int = 256):
        """
        PyTorch Dataset for Medical NER.
        Loads tokens and labels, tokenizes words, and aligns labels with subword tokens.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = Config.LABEL2ID
        
        logger.info(f"Loading dataset from {json_file_path}...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        logger.info(f"Loaded {len(self.data)} samples.")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        ner_tags = item["ner_tags"]
        
        # Build input_ids and label_ids word-by-word to support both fast and slow tokenizers
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [-100]
        
        for word, tag in zip(words, ner_tags):
            word_tokens = self.tokenizer.tokenize(word)
            word_sub_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            if not word_sub_ids:
                continue
                
            # First token gets the actual label ID, subsequent tokens get -100 (ignored in loss)
            tag_id = self.label2id.get(tag, 0)
            label_ids.append(tag_id)
            input_ids.append(word_sub_ids[0])
            
            for sub_id in word_sub_ids[1:]:
                label_ids.append(-100)
                input_ids.append(sub_id)
                
        # Truncate if necessary (leaving room for EOS token)
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[:self.max_len - 1]
            label_ids = label_ids[:self.max_len - 1]
            
        # Add EOS token
        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(-100)
        
        # Pad to max_len
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            label_ids.extend([-100] * padding_length)
            
        # Attention mask
        attention_mask = [1] * (self.max_len - padding_length) + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }
