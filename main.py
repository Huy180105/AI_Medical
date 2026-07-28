import argparse
import sys
import os
import uvicorn
import torch

# Ensure current working directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.generate_data import main as run_generate_data
from src.training.train_ner import train as run_train
from src.inference.predict_ner import MedicalNERPredictor

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Medical AI Agent - Vietnamese Clinical NLP CLI")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["generate-data", "train", "evaluate", "api", "predict"],
        help="Command execution mode"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Input text for NER prediction (required only for 'predict' mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate-data":
        logger.info("Executing Mode: Data Generation")
        run_generate_data()
        
    elif args.mode == "train":
        logger.info("Executing Mode: PhoBERT Fine-tuning")
        run_train()
        
    elif args.mode == "evaluate":
        logger.info("Executing Mode: Model Evaluation")
        # Load model and test dataset to run evaluation
        from transformers import AutoTokenizer
        from src.training.dataset import MedicalNERDataset
        from src.training.evaluate import evaluate_model_on_dataset
        
        test_path = os.path.join(Config.DATA_DIR, "test.json")
        if not os.path.exists(test_path):
            logger.error("Test dataset not found. Generating data first...")
            run_generate_data()
            
        logger.info("Loading predictor...")
        predictor = MedicalNERPredictor()
        
        logger.info("Loading test dataset...")
        test_dataset = MedicalNERDataset(test_path, predictor.tokenizer, max_len=Config.MAX_LEN)
        
        evaluate_model_on_dataset(
            model=predictor.model,
            tokenizer=predictor.tokenizer,
            dataset=test_dataset,
            device=predictor.device
        )
        
    elif args.mode == "api":
        logger.info("Executing Mode: Start FastAPI Server")
        uvicorn.run(
            "src.api.app:create_app", 
            host=Config.API_HOST, 
            port=Config.API_PORT, 
            factory=True,
            reload=False  # Keep false in production for stability
        )
        
    elif args.mode == "predict":
        logger.info("Executing Mode: Command Line Inference")
        if not args.text:
            parser.error("--text is required when --mode is 'predict'")
            
        logger.info(f"Input text: '{args.text}'")
        predictor = MedicalNERPredictor()
        entities = predictor.predict(args.text)
        
        print("\nExtracted Entities:")
        print("="*60)
        print(f"{'Text':<25} | {'Label':<12} | {'Confidence':<10} | {'Offsets':<10}")
        print("-"*60)
        for ent in entities:
            offsets = f"{ent['start']}:{ent['end']}"
            print(f"{ent['text']:<25} | {ent['type']:<12} | {ent['score']:<10.4f} | {offsets:<10}")
        print("="*60)

if __name__ == "__main__":
    main()
