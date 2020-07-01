import sys
import argparse
from tf_codage.utils import grep_keras_results_from_notebook


def main():
    """Grep jupyter notebook and create a table with training progress results."""

    parser = argparse.ArgumentParser()
    parser.add_argument("ipynb_file")
    parser.add_argument(
        "csv_output_file", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )
    parser.add_argument("--filter-by", default="val_loss")

    args = parser.parse_args()

    df = grep_keras_results_from_notebook(args.ipynb_file, filter_by=args.filter_by)

    df.to_csv(args.csv_output_file)
