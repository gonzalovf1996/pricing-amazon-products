from pathlib import Path
import pandas as pd


def save_df(
    obj: pd.DataFrame | pd.Series,
    path: str | Path,
    overwrite: bool = False,
    index: bool = False,
) -> None:
    """
    Save a DataFrame to parquet.

    Parameters
    ----------
    - obj (pd.DataFrame): Object to save.
    - path (str or Path): Output file path.
    - overwrite (bool): If False and the file exists, the function skips saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        print(f"Skipping {path.name} (already exists and overwrite mode set to False)")
        return

    if isinstance(obj, pd.Series):
        obj = obj.to_frame()

    obj.to_parquet(path, index=index)
    print(f"Saved {path.name}")