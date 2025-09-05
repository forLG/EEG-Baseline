import argparse
import sys
import os

# Import project root directory to sys.path to ensure correct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_baseline.api import train, evaluate

def main():
    parser = argparse.ArgumentParser(
        description="EEG Baseline Framework: A unified interface for training and evaluation."
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Train
    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the training YAML configuration file."
    )
    # TODO: Add more arguments to override config file settings if needed
    
    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    parser_eval.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the evaluation YAML configuration file."
    )
    parser_eval.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the saved model checkpoint (.pth file)."
    )

    args = parser.parse_args()

    # Based on the command, call the appropriate function
    if args.command == "train":
        train(config_path=args.config)
    elif args.command == "evaluate":
        evaluate(ckpt_path=args.checkpoint, config_path=args.config, )
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()