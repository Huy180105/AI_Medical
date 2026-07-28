import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from src.utils.config import Config
from src.utils.logger import get_logger
from src.training.dataset import MedicalNERDataset
from src.training.evaluate import compute_metrics

logger = get_logger(__name__)

def train():
    Config.print_config()
    
    # 1. Check GPU Setup
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Training will run on CPU, which is NOT recommended for PhoBERT.")

    # 2. Paths
    train_path = os.path.join(Config.DATA_DIR, "train.json")
    val_path = os.path.join(Config.DATA_DIR, "val.json")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("Dataset files not found. Please run data generation first.")
        raise FileNotFoundError("Missing train.json or val.json in data/ directory.")

    # 3. Load Tokenizer
    logger.info(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=False)

    # 4. Create Datasets
    logger.info("Preparing PyTorch Datasets...")
    train_dataset = MedicalNERDataset(train_path, tokenizer, max_len=Config.MAX_LEN)
    val_dataset = MedicalNERDataset(val_path, tokenizer, max_len=Config.MAX_LEN)

    # 5. Initialize Model
    logger.info(f"Loading pretrained PhoBERT model: {Config.MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID
    )
    
    # Send model to device
    model.to(Config.DEVICE)

    # 6. Define Training Arguments
    logger.info("Setting up TrainingArguments...")
    
    # Check if GPU is RTX 4060 for mixed precision (fp16) support
    fp16_supported = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.VALID_BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        logging_dir=os.path.join(Config.BASE_DIR, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16_supported,
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        report_to="none"  # Do not use wandb/tensorboard automatically
    )

    # Data Collator for Token Classification (handles padding dynamically)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 7. Initialize Trainer
    logger.info("Initializing Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 8. Run Fine-Tuning
    logger.info("Starting PhoBERT fine-tuning...")
    trainer.train()
    
    # 9. Save Best Model
    logger.info(f"Saving fine-tuned model and tokenizer to {Config.OUTPUT_MODEL_DIR}...")
    trainer.save_model(Config.OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_MODEL_DIR)
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    train()
