"""Repeat the unchanged board trial with completed-response file handoff."""
from . import run_trial
from .transport import CompleteResponseClient


def main():
    run_trial.PlantClient = CompleteResponseClient
    run_trial.main()


if __name__ == "__main__":
    main()
