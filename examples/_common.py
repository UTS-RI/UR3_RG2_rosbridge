"""Shared command-line options and figure output; processing stays in each example."""

import argparse
from pathlib import Path


def _overlay_alpha(value: str) -> float:
    try:
        alpha = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha must be a number between 0 and 1.") from exc
    if not 0 <= alpha <= 1:
        raise argparse.ArgumentTypeError("alpha must be between 0 and 1.")
    return alpha


def add_overlay_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=_overlay_alpha, default=0.5, metavar="0..1",
                        help="Depth overlay opacity: 0 = RGB, 1 = depth colours (default: 0.5)")


def example_args(description: str, *, overlay: bool = False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--demo", action="store_true", help="Synthetic data without a camera")
    parser.add_argument("--save", type=Path, metavar="IMAGE.png",
                        help="Save the figure instead of opening a window")
    if overlay:
        add_overlay_argument(parser)
    args = parser.parse_args()
    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    return args


def finish_figure(fig, path: Path | None) -> None:
    import matplotlib.pyplot as plt

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Image saved: {path}")
        plt.close(fig)
    else:
        plt.show()
