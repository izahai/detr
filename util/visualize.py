import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse
import os


def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_image(image_path):
    transform = T.Compose([
        T.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

def draw_bboxes(image, boxes, scores, labels, class_names=None):
    """
    Draw bounding boxes on the image, displaying confidence scores and class indices.
    
    Args:
    - image: The image on which to draw the boxes.
    - boxes: Bounding boxes as a numpy array of shape (N, 4).
    - scores: Confidence scores corresponding to each box.
    - labels: Class indices corresponding to each box.
    - class_names: A list of class names (optional), where the index corresponds to the label.
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)
    
    print("-----")
    for box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Display score and label (class index)
        label_text = f'Class: {label}'
        score_text = f'{score:.2f}'
        
        # If class names are provided, display the class name instead of the index
        #if class_names is not None:
         #   label_text = f'{class_names[label]}: {score:.2f}'
        
        ax.text(x_min, y_min, f'{label_text} {score_text}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.show()


def visualize_image(coco_data, image_dir, image_filename, image_width, image_height):
    # Build lookups
    filename_to_image = {img['file_name']: img for img in coco_data['images']}
    category_lookup = {cat['id']: cat['name'] for cat in coco_data['categories']}

    if image_filename not in filename_to_image:
        raise ValueError(f"Image {image_filename} not found in COCO JSON.")

    image_info = filename_to_image[image_filename]
    image_id = image_info['id']

    # Load the image using PIL for visualization
    image_path = os.path.join(image_dir, image_filename)
    _, image = load_image(image_path)

    # Filter annotations for this image
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Prepare bounding boxes, scores, and labels
    boxes = []
    scores = []  # In your case, you can set scores to 1 or load from other sources
    labels = []

    for ann in anns:
        # Extract bounding box details
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        label = category_lookup.get(category_id, "unknown")

        # Collect bounding box, score, and label
        boxes.append([x, y, x + w, y + h])  # [x_min, y_min, x_max, y_max]
        scores.append(1.0)  # Placeholder score, you can adjust this
        labels.append(label)

    # Draw bounding boxes on the image
    draw_bboxes(image, boxes, scores, labels, class_names=list(category_lookup.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO bounding boxes on an image")
    parser.add_argument('--json_path', type=str, required=True, help='Path to COCO-format JSON file (e.g., val.json)')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory where images are stored')
    parser.add_argument('--image_name', type=str, required=True, help='Filename of the image to visualize (e.g., 001.jpg)')
    parser.add_argument('--image_width', type=int, default=1280, help='Width of the image')
    parser.add_argument('--image_height', type=int, default=720, help='Height of the image')

    args = parser.parse_args()

    coco_data = load_coco_annotations(args.json_path)
    visualize_image(coco_data, args.image_dir, args.image_name, args.image_width, args.image_height)