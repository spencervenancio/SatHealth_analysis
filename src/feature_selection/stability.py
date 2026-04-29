from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.config import PROCESSED_DATA_DIR
from src.feature_selection.embedded import LASSO, MDI
from src.feature_selection.filter import corr_screening
from src.feature_selection.wrapper import loco

DROP_COLS = ["NDVI", "NDVI_binary", "NO2_column_number_density"]

app = typer.Typer()
console = Console()


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 1.0


def bootstrap_stability(method_fn, df, B=50, seed=42):
    rng = np.random.RandomState(seed)
    selections = [
        list(method_fn(df.iloc[rng.choice(len(df), size=len(df), replace=True)].reset_index(drop=True)))
        for _ in range(B)
    ]
    scores = [
        jaccard(selections[i], selections[j])
        for i in range(len(selections))
        for j in range(i + 1, len(selections))
    ]
    return float(np.mean(scores))


@app.command()
def main(
    top_k: int = typer.Option(10, help="Number of features to select per method."),
    target_col: str = typer.Option("depr_prev", help="Target column name."),
    data_path: Optional[Path] = typer.Option(None, help="Override default dataset path."),
    B: int = typer.Option(50, "--B", help="Number of bootstrap iterations."),
    include_loco: bool = typer.Option(False, "--include-loco", help="Include LOCO (very slow)."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
):
    path = data_path or PROCESSED_DATA_DIR / "dataset.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    methods = [
        ("corr_screening", lambda d: corr_screening(d, target_col=target_col, selection_criterion="top_K", top_K_features=top_k)),
        ("MDI",            lambda d: MDI(d, target_col=target_col, top_k=top_k)),
        ("LASSO",          lambda d: LASSO(d, target_col=target_col, top_k=top_k)),
    ]
    if include_loco:
        methods.append(("LOCO", lambda d: loco(d, target_col=target_col, top_k=top_k)))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Method")
    table.add_column(f"Jaccard (B={B})", justify="right")

    for name, fn in methods:
        console.print(f"[yellow]{name}...[/yellow]")
        score = bootstrap_stability(fn, df, B=B, seed=seed)
        table.add_row(name, f"{score:.3f}")

    console.print(table)


if __name__ == "__main__":
    app()
