"""VLCS sweep: python -m experiments.delayed_feedback.run_vlcs --gpus 0 1."""

from .run import main


if __name__ == "__main__":
    main(default_dataset="VLCS")
