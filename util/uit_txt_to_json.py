import json
import os
import argparse

# Define the class labels
labels = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet"]
label_to_id = {name: i + 1 for i, name in enumerate(labels)}  # COCO uses 1-indexed category IDs

# Function to convert data into COCO format
def convert_to_coco(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
        "info": {"description": "Object Detection Dataset"}
    }

    # Add categories to the COCO format based on the labels
    for i, label in enumerate(labels, 1):  # Labels are 1-indexed in COCO format
        coco_data['categories'].append({
            "id": i,
            "name": label,
            "supercategory": "none"
        })

    # Process the file
    image_id = 1
    annotation_id = 1
    for line in lines:
        # Split each line by commas
        parts = line.strip().split(',')

        # Frame number and other info
        video_id = parts[0]  # Video ID, can be used for naming frames
        frame_num = int(parts[1])  # Frame number
        x = float(parts[2])  # X-coordinate
        y = float(parts[3])  # Y-coordinate
        width = float(parts[4])  # Width of the bounding box
        height = float(parts[5])  # Height of the bounding box
        object_id = int(float(parts[6]))  # Object ID (this is the class ID)
        confidence = float(parts[7])  # Confidence score (not used here)

        # Get the label for the object based on object_id
        if object_id in label_to_id.values():
            category_id = object_id
        else:
            print(f"Warning: Object ID {object_id} is not in the predefined label set.")
            category_id = 0  # Default to 0 if the object_id doesn't match any label

        # Image file name
        image_filename = f"{str(frame_num).zfill(3)}_{video_id}.jpg"

        # Add image data
        coco_data['images'].append({
            "id": image_id,
            "width": 1280,  # You may need to adjust these values
            "height": 720,  # Adjust based on your video resolution
            "file_name": image_filename
        })

        # Add annotation (bounding box)
        coco_data['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,  # Use category_id based on object_id
            "bbox": [x, y, width, height],
            "area": width * height,  # Optional but commonly used in COCO
            "iscrowd": 0
        })

        # Increment IDs
        image_id += 1
        annotation_id += 1

    # Save the result as a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(coco_data, json_file, indent=2)


# Main function to parse arguments and run the script
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert bounding box annotations to COCO format")
    parser.add_argument('--input_file', type=str, help="Path to the input text file with annotations")
    parser.add_argument('--output_file', type=str, help="Path to save the output COCO JSON file")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the file to COCO format
    convert_to_coco(args.input_file, args.output_file)
    print(f"COCO JSON file saved as {args.output_file}")


if __name__ == "__main__":
    main()