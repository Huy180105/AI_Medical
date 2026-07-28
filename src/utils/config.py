import os

try:
    import torch
except ImportError:
    torch = None

class Config:
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "vinai/phobert-base")
    
    # Path Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_MODEL_DIR = os.path.join(MODELS_DIR, "phobert-medical-ner")
    KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", os.path.join(BASE_DIR, "knowledge_base"))
    VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", os.path.join(KNOWLEDGE_BASE_DIR, "faiss.index"))
    VECTOR_METADATA_PATH = os.getenv("VECTOR_METADATA_PATH", os.path.join(KNOWLEDGE_BASE_DIR, "metadata.pkl"))
    
    # Training Hyperparameters
    MAX_LEN = int(os.getenv("MAX_LEN", 256))
    TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 8))
    VALID_BATCH_SIZE = int(os.getenv("VALID_BATCH_SIZE", 8))
    EPOCHS = int(os.getenv("EPOCHS", 5))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.1))
    
    # Hardware Configuration
    DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 5))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    AGENT_MEMORY_TTL_SECONDS = int(os.getenv("AGENT_MEMORY_TTL_SECONDS", 86400))
    
    # MLOps Configuration
    EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", os.path.join(BASE_DIR, "experiments"))
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", os.path.join(BASE_DIR, "mlruns"))
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "medical-ai-agent")
    MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "phobert-medical-ner")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "medical-ai-agent")
    WANDB_MODE = os.getenv("WANDB_MODE", "offline")
    POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://mlflow:mlflow@postgres:5432/mlflow")
    METRICS_LATENCY_PROBE_URL = os.getenv("METRICS_LATENCY_PROBE_URL", "http://localhost:8000/health")
    
    # NER Labels Mapping
    # Standard BIO tagging format for NER
    # Entities: SYMPTOM, DISEASE, MEDICINE, TEST
    LABEL_LIST = [
        "O",
        "B-SYMPTOM", "I-SYMPTOM",
        "B-DISEASE", "I-DISEASE",
        "B-MEDICINE", "I-MEDICINE",
        "B-TEST", "I-TEST"
    ]
    
    LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
    NUM_LABELS = len(LABEL_LIST)
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    @classmethod
    def print_config(cls):
        print("="*50)
        print("MEDICAL AI AGENT CONFIGURATION")
        print(f"Model Name: {cls.MODEL_NAME}")
        print(f"Data Dir: {cls.DATA_DIR}")
        print(f"Model Save Dir: {cls.OUTPUT_MODEL_DIR}")
        print(f"Knowledge Base Dir: {cls.KNOWLEDGE_BASE_DIR}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
        print(f"MLflow URI: {cls.MLFLOW_TRACKING_URI}")
        print(f"Experiments Dir: {cls.EXPERIMENTS_DIR}")
        print(f"Device: {cls.DEVICE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.TRAIN_BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Labels: {cls.LABEL_LIST}")
        print("="*50)
