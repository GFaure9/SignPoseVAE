import argparse
from .evaluation.latent_eval import evaluate_latent_space


def parse_list(arg):
    """Convert comma-separated CLI argument to list."""
    if arg is None:
        return None
    return [x.strip() for x in arg.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent space statistics")

    parser.add_argument(
        "--output_folders",
        type=str,
        required=True,
        help="Comma separated list of output folders"
    )

    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma separated list of experiment names"
    )

    parser.add_argument(
        "--cfg_model_filepaths",
        type=str,
        required=True,
        help="Comma separated list of model config file paths"
    )

    parser.add_argument(
        "--ckpt_filepaths",
        type=str,
        required=True,
        help="Comma separated list of saved model checkpoints (.ckpt) file paths - must match models order"
    )

    parser.add_argument(
        "--cfg_data_filepaths",
        type=str,
        required=True,
        help="Comma separated list of dataset config file paths"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma separated list of metrics"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--plot_correl",
        action="store_true",
        help="Plot correlation matrices"
    )

    args = parser.parse_args()

    evaluate_latent_space(
        output_folders=parse_list(args.output_folders),
        names=parse_list(args.names),
        cfg_model_filepaths=parse_list(args.cfg_model_filepaths),
        ckpt_filepaths=parse_list(args.ckpt_filepaths),
        cfg_data_filepaths=parse_list(args.cfg_data_filepaths),
        metrics=parse_list(args.metrics),
        batch_size=args.batch_size,
        plot_correl=args.plot_correl,
    )


if __name__ == "__main__":
    main()
