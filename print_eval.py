import torch
import argparse
from pycocotools.cocoeval import Params  # Import the Params class to allow safe loading

def load_and_print_map(eval_file):
    """
    Load the eval.pth file and print the mAP score.

    Args:
    - eval_file (str): Path to the eval.pth file.
    """
    try:
        # Allow the Params class from pycocotools to be safely loaded
        torch.serialization.add_safe_globals([Params])

        # Load the eval.pth file with weights_only=False
        eval_data = torch.load(eval_file, weights_only=False)

        # Extract and print mAP scores
        if isinstance(eval_data, dict) and "bbox" in eval_data:
            bbox_eval = eval_data["bbox"]
            if hasattr(bbox_eval, "stats"):
                print(f"mAP (IoU=0.50:0.95): {bbox_eval.stats[0]:.4f}")
                print(f"mAP (IoU=0.50): {bbox_eval.stats[1]:.4f}")
                print(f"mAP (IoU=0.75): {bbox_eval.stats[2]:.4f}")
            else:
                print("Error: 'stats' attribute not found in bbox evaluation data.")
        else:
            print("Error: 'bbox' key not found in eval data or invalid format.")
    except FileNotFoundError:
        print(f"Error: File {eval_file} not found.")
    except Exception as e:
        print(f"An error occurred while loading {eval_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print the mAP score from an eval.pth file.")
    parser.add_argument('--eval_file', type=str, required=True, help="Path to the eval.pth file.")
    args = parser.parse_args()

    load_and_print_map(args.eval_file)