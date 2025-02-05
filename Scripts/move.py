import os
import shutil

def move_files(src_folder, dest_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
        print(f"Created destination folder: {dest_folder}")
    
    # Walk through all directories and subdirectories in the source folder
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # Get full file path
            file_path = os.path.join(root, file)
            try:
                # Move the file to the destination folder
                shutil.move(file_path, dest_folder)
                print(f"Moved file: {file_path} to {dest_folder}")
            except Exception as e:
                print(f"Error moving file {file_path} to {dest_folder}: {e}")
    
    print(f"All files from {src_folder} moved to {dest_folder}.")

# Example usage
src_folder = 'C:/Users/jesli/Downloads/synthetic_data'
dest_folder = 'GAN-YOLO-Detction/gan_data'

move_files(src_folder, dest_folder)
