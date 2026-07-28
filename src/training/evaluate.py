import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_metrics(p):
    """
    Computes NER metrics (F1, Precision, Recall, Accuracy) for Hugging Face Trainer evaluation.
    p: tuple of (predictions, labels)
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index -100 (special tokens) and map ids to labels
    true_predictions = [
        [Config.ID2LABEL[p_id] for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [Config.ID2LABEL[l_id] for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions)
    }
    
    return results

def evaluate_model_on_dataset(model, tokenizer, dataset, device="cuda"):
    """
    Evaluates a PyTorch model on a PyTorch dataset and prints a detailed classification report.
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    logger.info("Starting model evaluation...")
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            
            predictions = np.argmax(logits, axis=2)
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(labels_np)
            
    # Align labels and prediction lists, omitting -100
    true_preds = [
        [Config.ID2LABEL[p_id] for (p_id, l_id) in zip(pred, label) if l_id != -100]
        for pred, label in zip(all_preds, all_labels)
    ]
    true_targets = [
        [Config.ID2LABEL[l_id] for (p_id, l_id) in zip(pred, label) if l_id != -100]
        for pred, label in zip(all_preds, all_labels)
    ]
    
    # Calculate scores
    p = precision_score(true_targets, true_preds)
    r = recall_score(true_targets, true_preds)
    f1 = f1_score(true_targets, true_preds)
    acc = accuracy_score(true_targets, true_preds)
    
    logger.info("="*50)
    logger.info("EVALUATION METRICS SUMMARY")
    logger.info(f"Precision: {p:.4f}")
    logger.info(f"Recall:    {r:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info("="*50)
    
    report = classification_report(true_targets, true_preds)
    print("\nDetailed Classification Report:")
    print(report)
    
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,
        "report": report
    }
