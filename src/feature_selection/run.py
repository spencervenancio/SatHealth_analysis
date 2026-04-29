from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


@app.command()
def main(
    top_k: int = typer.Option(10, help="Number of features to select per method."),
    target_col: str = typer.Option("depr_prev", help="Target column name."),
    data_path: Optional[Path] = typer.Option(None, help="Override default dataset path."),
    compare: bool = typer.Option(False, "--compare", help="Print a feature x method comparison table and correlation matrix."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Export the --compare table to a CSV and correlation matrix to a PNG."),
):
    path = data_path or PROCESSED_DATA_DIR / "dataset.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    methods = [
        ("corr_screening", "Filter",   lambda: corr_screening(df, target_col=target_col, selection_criterion="top_K", top_K_features=top_k)),
        ("MDI",            "Embedded", lambda: MDI(df, target_col=target_col, top_k=top_k)),
        ("LASSO",          "Embedded", lambda: LASSO(df, target_col=target_col, top_k=top_k)),
        ("LOCO",           "Wrapper",  lambda: loco(df, target_col=target_col, top_k=top_k)),
    ]

    results = {}
    for name, _, fn in methods:
        if name == "LOCO":
            console.print(f"[yellow]Running LOCO (slow — fits {df.shape[1]-1} models)...[/yellow]")
        results[name] = set(fn().tolist())

    if compare:
        selected_by_any = sorted(set.union(*results.values()))

        if output is not None:
            export = pd.DataFrame(
                {name: [int(f in results[name]) for f in selected_by_any] for name in results},
                index=selected_by_any,
            )
            export.index.name = "feature"
            export.to_csv(output)
            console.print(f"[green]Saved comparison table to {output}[/green]")

        # Correlation matrix for the union of selected features
        corr = df[selected_by_any].corr()
        n = len(selected_by_any)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title(f"Correlation matrix — union of top-{top_k} features ({n} total)")
        plt.tight_layout()
        if output is not None:
            plot_path = output.with_suffix(".png")
            fig.savefig(plot_path, dpi=150)
            console.print(f"[green]Saved correlation plot to {plot_path}[/green]")
        else:
            plt.show()

        table = Table(show_header=True, header_style="bold")
        table.add_column("Feature")
        for name in results:
            table.add_column(name, justify="center")
        for feature in selected_by_any:
            table.add_row(feature, *["[green]✓[/green]" if feature in results[name] else "[dim]·[/dim]" for name in results])
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Method")
        table.add_column("Category")
        table.add_column("N", justify="right")
        table.add_column("Features")
        for (name, category, _), features in zip(methods, results.values()):
            table.add_row(name, category, str(len(features)), ", ".join(sorted(features)))

    console.print(table)


if __name__ == "__main__":
    app()
