import os
    

def fix_missing_frames(folder_path, typee):
    for i in range(1,101):
        video_str = f"{i:03}"
        ind = 1
        for j in range(1, 51):
            file_name = f"{video_str}_{j}."+ typee
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as f:
                    pass
                new_file_name = f"{video_str}_{ind}.{typee}"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")
                ind += 1  # Increment the index for the next file
            except FileNotFoundError:
                print(f"File not found: {file_name}, skipping...")

if __name__ == "__main__":
    folder_path = "images"  # Replace with the path to your folder containing .txt files
    fix_missing_frames(folder_path, "jpg")