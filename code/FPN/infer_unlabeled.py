import os
import glob
import cv2
import torch
import numpy as np
import argparse
from seg_head import SegModel

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Pretrained Segmentor (S)
    print(f"Loading pretrained weights from: {args.checkpoint}")
    model = SegModel(backbone_name=args.backbone, pretrained=False, num_classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Collect unlabeled tissue images
    valid_exts = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
    tissue_files = sorted([
        f for f in glob.glob(os.path.join(args.tissue_dir, "**", "*.*"), recursive=True)
        if f.lower().endswith(valid_exts)
    ])

    if len(tissue_files) == 0:
        raise FileNotFoundError(f"No image files found in '{args.tissue_dir}'")

    print(f"Found {len(tissue_files)} unlabeled tissue images. Processing first {args.num_samples}...")

    # 3. Predict masks on unlabeled samples
    with torch.no_grad():
        for i, img_path in enumerate(tissue_files[:args.num_samples]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, tuple(args.target_size), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0.0, 1.0] matching pretraining input
            img_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
            
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (probs > args.threshold).astype(np.uint8) * 255

            # Save binary mask
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_save_path = os.path.join(args.output_dir, f"{base_name}_pred_mask.png")
            cv2.imwrite(mask_save_path, pred_bin)

            # Save side-by-side visual comparison (Raw Image + Binary Mask + Green Overlay)
            overlay = img_resized.copy()
            overlay[pred_bin > 0] = [0, 255, 0]  # Green overlay
            blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
            
            comparison = np.hstack([
                cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            ])
            comp_save_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
            cv2.imwrite(comp_save_path, comparison)

    print(f"Generated masks saved in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue-dir", default="WSI_Patches")
    parser.add_argument("--checkpoint", default="checkpoints/pretrain/resnet50_stage2_pretrained.pth")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--output-dir", default="data/unlabeled_predictions")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--target-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--threshold", type=float, default=0.5)
    main(parser.parse_args())