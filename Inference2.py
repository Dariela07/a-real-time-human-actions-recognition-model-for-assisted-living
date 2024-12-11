import argparse
from mmaction.apis import inference_recognizer, init_recognizer
import logging
from mmengine.logging import MMLogger

# Suppress warnings and logs
logging.getLogger().setLevel(logging.ERROR)
MMLogger.get_instance("mmengine").setLevel(logging.ERROR)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Video action recognition using MMAction2.")
    parser.add_argument("config_path", type=str, help="Path to the model configuration file.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file.")
    parser.add_argument("img_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()

    # Model initialization
    model = init_recognizer(args.config_path, args.checkpoint_path, device="cuda:0")

    # Perform inference
    result = inference_recognizer(model, args.img_path)

    # Process results
    pred_label = result.pred_label.item()
    classes = ['Falling', 'Staggering', 'Chest Pain', "Normal"]

    print("Predicted Class:", classes[pred_label])

if __name__ == "__main__":
    main()
