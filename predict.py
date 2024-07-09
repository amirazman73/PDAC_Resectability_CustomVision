from Models.setup import setup_model
import torch
import pandas as pd
from Data.preprocess import MedicalImageDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Data.data_loader import load_labels, setup_dir
from Data.utils import find_max_bounding_box_edges


raw_ct_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
seg_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'
label_dir = r'.../GitHub/PDAC_Resectability_CustomVision/Data/files'


labels_df = load_labels(label_dir)
seg_dir = setup_dir(seg_dir, raw_ct_dir=raw_ct_dir, segmentations_dir=seg_dir)

train_data, test_data = train_test_split(labels_df, test_size=0.1, random_state=42)

max_height, max_width, max_depth = find_max_bounding_box_edges(seg_dir)

model = setup_model()
model_path = r'.../GitHub/PDAC_Resectability_CustomVision/Data/saved_states/Final_All_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.to(device)

test_dataset = MedicalImageDataset(test_data, raw_ct_dir, seg_dir, max_height, max_width, max_depth)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_predictions = []
all_true_labels = []

with torch.no_grad():
    for batch_data in test_loader:
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

predictions_df = pd.DataFrame({
    'TrueLabel': all_true_labels,
    'PredictedLabel': all_predictions
})

# print(predictions_df)
      
correct_predictions = (predictions_df['TrueLabel'] == predictions_df['PredictedLabel']).sum()
total_predictions = len(predictions_df)
accuracy = (correct_predictions / total_predictions) * 100

print(f"Accuracy: {accuracy:.2f}%")