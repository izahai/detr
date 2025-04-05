import os
import argparse

def concat_txt_files(input_folder, output_file):
    """
    Concatenate all .txt files in a folder into a single .txt file.

    Args:
    - input_folder (str): Path to the folder containing .txt files.
    - output_file (str): Path to the output .txt file.
    """
    with open(output_file, 'w') as outfile:
        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
    print(f"All .txt files in '{input_folder}' have been concatenated into '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate all .txt files in a folder into a single .txt file.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder containing .txt files.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output .txt file.")
    args = parser.parse_args()

    concat_txt_files(args.input_folder, args.output_file)