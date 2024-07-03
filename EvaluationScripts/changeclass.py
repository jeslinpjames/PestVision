import os

def change_class_id_to_zero(input_dir):
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            
            # Read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Change the class ID to 0 for each line
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                parts[0] = '0'
                new_lines.append(' '.join(parts))
            
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.write('\n'.join(new_lines))
    
    print("Class IDs have been changed to 0 for all text files in the directory.")

# Define the input directory containing the text files
input_dir = r"D:/git/Data/PestVisionEvalImages/Label"

# Change class IDs to 0
change_class_id_to_zero(input_dir)
