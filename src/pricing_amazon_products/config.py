from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Evaluation metric
EVAL_METRIC  = mean_squared_error
SCORING = "neg_root_mean_squared_error"
TARGET = "log_price"
N_FOLDS = 5
RANDOM_STATE = 42

# Embeddings models
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "max_tokens": 256,
        "weight_in_million_params": 23,
        "dimensions": 384,
        "mteb_position": 133,
    },
    "avsolatorio/GIST-small-Embedding-v0": {
        "max_tokens": 512,
        "weight_in_million_params": 33,
        "dimensions": 384,
        "mteb_position": 50,
    },
}
EMBEDDING_SOURCE_COL = "item_name"

# PATHS
DATA_DIR = Path("../../data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SPLITS_DIR = DATA_DIR / "splits"
LLM_PREDICTIONS_DIR = DATA_DIR / "llm_predictions"
MODELS_METADATA_PATH = Path("../../research/experiments/model_metadata.parquet")
MODELS_DIR = DATA_DIR / "models"

