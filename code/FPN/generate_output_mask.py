import torch
import cv2
import numpy as np
import os
from seg_head import SegModel

def process_set(model, img_dir, mask_dir, img_list, source_label, device, output_folder):
    """Helper to process images and save with a 10px gap between panels."""
    for img_name in img_list:
        # --- Paths ---
        img_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_bin_mask.png"
        mask_path = os.path.join(mask_dir, mask_name)

        # --- Load and Preprocess ---
        original_bgr = cv2.imread(img_path)
        if original_bgr is None: continue
        h, w = original_bgr.shape[:2]
        
        img_input = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (512, 512))
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        gt_mask = cv2.imread(mask_path, 0)
        gt_mask = np.zeros((h, w), dtype=np.uint8) if gt_mask is None else (gt_mask > 0).astype(np.uint8) * 255

        with torch.no_grad():
            pred = model(img_tensor)
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        pred_mask = (prob > 0.5).astype(np.uint8) * 255
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- Arrange Triple-Pane Visualization ---
        display_size = (512, 512)
        pane1 = cv2.resize(original_bgr, display_size)
        pane2 = cv2.resize(cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR), display_size)
        pane3 = cv2.resize(cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR), display_size)

        # 1. Create a 10-pixel wide black vertical separator
        gap_width = 10
        gap = np.zeros((display_size[0], gap_width, 3), dtype=np.uint8) 
        # (Optional: Use 255 for a white gap)

        # 2. Stack images with the gap in between
        combined = np.hstack((pane1, gap, pane2, gap, pane3))

        # --- Add Labels ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) # Green text
        
        # Adjust text coordinates slightly to account for the new gaps
        cv2.putText(combined, f"Source: {source_label}", (10, 490), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Original", (10, 35), font, 1, color, 2)
        cv2.putText(combined, "Ground Truth", (512 + gap_width + 10, 35), font, 1, color, 2)
        cv2.putText(combined, "FPN+DWT Prediction", (1024 + 2*gap_width + 10, 35), font, 1, color, 2)

        # --- Save ---
        save_path = os.path.join(output_folder, f"{source_label}_{base_name}.png")
        cv2.imwrite(save_path, combined)
        print(f"Saved: {save_path}")
def main():
    # Paths
    train_imgs = "../MonuSeg/MonuSeg/Training/TissueImages"
    train_masks = "../MonuSeg/MonuSeg/Training/GroundTruth"
    test_imgs = "../MonuSeg/MonuSeg/Test/TissueImages"
    test_masks = "../MonuSeg/MonuSeg/Test/GroundTruth"
    model_path = "checkpoints/fpn_monuseg_final.pth"
    output_dir = "output"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    model = SegModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get file lists
    ext = ('.png', '.tif')
    train_list = [f for f in os.listdir(train_imgs) if f.lower().endswith(ext)][:5]
    test_list = [f for f in os.listdir(test_imgs) if f.lower().endswith(ext)][:5]

    # Run Process
    process_set(model, train_imgs, train_masks, train_list, "TRAIN_SET", device, output_dir)
    process_set(model, test_imgs, test_masks, test_list, "TEST_SET", device, output_dir)

if __name__ == "__main__":
    main()