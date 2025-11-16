"""
bertscore_tester.py
Compute semantic similarity between reference and generated summaries using BERTScore.
"""

from bert_score import score as bert_score
from rich.console import Console
from rich.table import Table
import os


class BERTScoreTester:
    """
    A class to compute semantic similarity between reference and candidate texts
    using BERTScore (default model: RoBERTa-Large).
    """

    def __init__(self, model: str = "roberta-large", verbose: bool = True):
        """
        Initialize the tester with a given model.

        Args:
            model (str): Hugging Face model name (default: 'roberta-large')
            verbose (bool): Whether to show detailed logs
        """
        self.model = model
        self.verbose = verbose
        self.console = Console()

    def compute(self, c: str, r: str) -> dict:
        """
        Compute BERTScore between two text strings.

        Args:
            c (str): Candidate (generated) text.
            r (str): Reference (ground truth) text.

        Returns:
            dict: { "precision": float, "recall": float, "f1": float }
        """
        if self.verbose:
            self.console.log(f"[bold cyan]Loading BERT model:[/bold cyan] {self.model}")

        P, R, F1 = bert_score(
            [c],
            [r],
            model_type=self.model,
            verbose=self.verbose,
        )

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }

    def compute_from_files(self, r_path: str, c_path: str) -> dict:
        """
        Compute BERTScore from two text files.

        Args:
            r_path (str): Path to reference text file.
            c_path (str): Path to candidate text file.

        Returns:
            dict: Computed BERTScore metrics.
        """
        if not os.path.exists(r_path) or not os.path.exists(c_path):
            raise FileNotFoundError("Reference or candidate file not found.")

        with open(r_path, "r", encoding="utf-8") as ref_file:
            r = ref_file.read()
        with open(c_path, "r", encoding="utf-8") as cand_file:
            c = cand_file.read()

        return self.compute(c, r)

    def pretty_print(self, scores: dict):
        """
        Display BERTScore metrics in a formatted Rich table.

        Args:
            scores (dict): Dictionary of precision, recall, and F1.
        """
        table = Table(title="BERTScore Evaluation")
        table.add_column("Metric", style="cyan", justify="center")
        table.add_column("Score", style="green", justify="center")

        for k, v in scores.items():
            table.add_row(k.capitalize(), f"{v:.4f}")

        self.console.print(table)


# ------------------------------
# CLI entry point
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute BERTScore between two text files.")
    parser.add_argument("-r", required=True, help="Path to reference (ground truth) text file")
    parser.add_argument("-c", required=True, help="Path to candidate (generated) text file")
    parser.add_argument("--model", default="roberta-large", help="Hugging Face model name")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs")
    args = parser.parse_args()

    tester = BERTScoreTester(model=args.model, verbose=not args.quiet)
    scores = tester.compute_from_files(r_path=args.r, c_path=args.c)
    tester.pretty_print(scores)
