import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from sentence_transformers import SentenceTransformer, models
    print("Loading PhoBERT Base from local cache...")
    word_embedding_model = models.Transformer('vinai/phobert-base')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    print("Loaded PhoBERT Base successfully!")
    
    # Compute embeddings for test sentences
    sentences = ["tăng huyết áp", "cao huyết áp", "tiểu đường"]
    embeddings = model.encode(sentences)
    print(f"Generated embeddings shape: {embeddings.shape}")
except Exception as e:
    print("Failed to load PhoBERT Base offline:", e)
