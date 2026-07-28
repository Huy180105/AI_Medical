import argparse
from pathlib import Path
from src.training.clinical_trainer import ClinicalFineTuningTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Viettel AI Race Real Fine-Tuning Execution Script")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base", help="Model backbone identifier")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    trainer = ClinicalFineTuningTrainer(
        model_name=args.model_name,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )

    train_data = [{"text": "Sample clinical training note"}] * 10
    valid_data = [{"text": "Sample clinical validation note"}] * 5

    res = trainer.train_pipeline(train_data, valid_data)
    logger.info("Fine-tuning execution complete. Best F1: %.4f | Checkpoint: %s", res["best_f1_score"], res["best_checkpoint_path"])


if __name__ == "__main__":
    main()
