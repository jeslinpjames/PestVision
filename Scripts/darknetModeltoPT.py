import torch
import sys
from collections import OrderedDict

# Add the path to your Darknet repository
darknet_path = 'D:/git/Darknet/yolov4/darknet'
sys.path.append(darknet_path)

# Import the darknet2pytorch from the tool module
from tool import darknet2pytorch

# Path to your YOLOv4 config and weights
cfg_file = "D:/git/Darknet/yolov4/darknet/cfg/yolov4-custom.cfg"
weights_file = "D:/git/Darknet/yolov4/training/yolov4-custom_4000.weights"
output_file = 'yolov4-pytorch-corrected.pt'

# Load weights from darknet format
model = darknet2pytorch.Darknet(cfg_file, inference=True)
model.load_weights(weights_file)

# Function to rename the keys in the state dictionary
def rename_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Modify the key names to match the Ultralytics YOLO model format
        new_key = key.replace('module.', '')  # Example of a key modification
        new_key = new_key.replace('conv', 'model.conv')  # This may need adjustment based on actual layer names
        new_key = new_key.replace('bn', 'model.bn')  # This may need adjustment based on actual layer names
        new_state_dict[new_key] = value
    return new_state_dict

# Rename the keys in the state dictionary
corrected_state_dict = rename_state_dict_keys(model.state_dict())

# Create a checkpoint dictionary with the expected keys
ckpt = {'model': corrected_state_dict, 'epoch': 0, 'cfg': cfg_file}

# Save the checkpoint dictionary to the .pt file
torch.save(ckpt, output_file)
