from pathlib import Path
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings

from pricing_amazon_products.io import save_df


def build_embedding_df(
    df_features: pd.DataFrame,
    col: str,
    model_name: str,
    emb_dir: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    emb_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.replace("/", "_")
    emb_path = emb_dir / f"{col}__{safe_model_name}.parquet"

    if emb_path.exists() and not overwrite:
        print(f"Skipping {emb_path.name} (already exists and overwrite is False)")
        return 

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    texts = df_features[col].fillna("").astype(str).tolist()
    vectors = embeddings.embed_documents(texts)

    df_embedding = pd.DataFrame(
        vectors,
        columns=[f"{col}_emb_{i}" for i in range(len(vectors[0]))]
    )
    df_embedding.insert(0, "row_id", df_features["row_id"].values)

    save_df(df_embedding, emb_path, overwrite=overwrite)
    return 