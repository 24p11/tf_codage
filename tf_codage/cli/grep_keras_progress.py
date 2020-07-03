import sys
import argparse
from tf_codage.utils import grep_keras_results_from_notebook


def get_parser():

    parser = argparse.ArgumentParser(
        description="Parse jupyter notebook and create a table with training progress results."
    )
    parser.add_argument("ipynb_file")
    parser.add_argument(
        "csv_output_file", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )
    parser.add_argument("--filter-by", default="val_loss")

    return parser


parser = get_parser()


def main():

    args = parser.parse_args()

    df = grep_keras_results_from_notebook(args.ipynb_file, filter_by=args.filter_by)

    df.to_csv(args.csv_output_file)
