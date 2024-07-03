import xml.etree.ElementTree as ET
import os
from multiprocessing import Pool
from tqdm import tqdm

# Define a dictionary to map class names to integer IDs
class_mapping = {
    'pest': 0,
    # Add other class name mappings here if necessary
}

def convert_to_yolo(xml_file, output_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract image dimensions
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    # Open the output file
    with open(output_file, 'w') as f:
        # Iterate over all objects in the XML file
        for obj in root.findall('object'):
            # Extract the class name and bounding box coordinates
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue  # Skip unknown classes
            
            class_id = class_mapping[class_name]
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            # Calculate the YOLO format coordinates
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / float(width)
            bbox_height = (ymax - ymin) / float(height)
            
            # Write the YOLO format line to the file
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def process_file(args):
    xml_file, output_file = args
    convert_to_yolo(xml_file, output_file)

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a list of files to process
    files_to_process = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.xml'):
            xml_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace('.xml', '.txt'))
            files_to_process.append((xml_file, output_file))
    
    # Use multiprocessing to process files
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process), desc="Processing files"))

if __name__ == "__main__":
    # Example usage
    input_folder = 'C:/Users/jesli/Downloads/j_scrap/xml/'
    output_folder = 'C:/Users/jesli/Downloads/j_scrap/yolo/'
    process_folder(input_folder, output_folder)
