import argparse
import os
import cv2
import numpy as np
import torch
from seg_head import SegModel


def find_mask_path(mask_dir, base_name):
  """Locates the binary mask using common naming conventions."""
  candidates = [
      os.path.join(mask_dir, f"{base_name}_bin_mask.png"),
      os.path.join(mask_dir, f"{base_name}.png"),
      os.path.join(mask_dir, f"{base_name}_mask.png"),
      os.path.join(mask_dir, f"{base_name}_bin_mask.tif"),
      os.path.join(mask_dir, f"{base_name}.tif"),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


def process_set(
    model,
    img_dir,
    mask_dir,
    img_list,
    source_label,
    device,
    output_folder,
    target_size=(512, 512),
    display_size=(512, 512),
    threshold=0.5,
    title_tag="UNITPath Prediction",
):
  """Processes images and generates triple-pane qualitative comparison panels."""
  valid_count = 0
  for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    base_name = os.path.splitext(img_name)[0]

    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
      continue
    h, w = original_bgr.shape[:2]

    # Preprocess RGB input
    img_input = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(
        img_input, tuple(target_size), interpolation=cv2.INTER_LINEAR
    )
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Locate ground truth mask
    mask_path = find_mask_path(mask_dir, base_name)
    if mask_path is not None:
      gt_mask = cv2.imread(mask_path, 0)
      gt_mask = (gt_mask > 0).astype(np.uint8) * 255
    else:
      gt_mask = np.zeros((h, w), dtype=np.uint8)

    # Model inference
    with torch.no_grad():
      pred = model(img_tensor)
      prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Binarize and resize mask back to source aspect ratio
    pred_mask = (prob > threshold).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Format visualization panes
    disp_w, disp_h = display_size
    pane1 = cv2.resize(original_bgr, (disp_w, disp_h))
    pane2 = cv2.resize(
        cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR), (disp_w, disp_h)
    )
    pane3 = cv2.resize(
        cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR), (disp_w, disp_h)
    )

    gap_width = 10
    gap = np.zeros((disp_h, gap_width, 3), dtype=np.uint8)
    combined = np.hstack((pane1, gap, pane2, gap, pane3))

    # Text overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    cv2.putText(
        combined,
        f"Source: {source_label}",
        (10, disp_h - 20),
        font,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(combined, "Original", (10, 35), font, 0.9, color, 2)
    cv2.putText(
        combined,
        "Ground Truth",
        (disp_w + gap_width + 10, 35),
        font,
        0.9,
        color,
        2,
    )
    cv2.putText(
        combined,
        title_tag,
        (2 * disp_w + 2 * gap_width + 10, 35),
        font,
        0.9,
        color,
        2,
    )

    save_path = os.path.join(output_folder, f"{source_label}_{base_name}.png")
    cv2.imwrite(save_path, combined)
    print(f"Saved: {save_path}")
    valid_count += 1

  return valid_count


def main():
  parser = argparse.ArgumentParser(
      description="Generate triple-pane qualitative prediction panels"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default="checkpoints/final/resnet50_best.pth",
      help="Path to trained model weights",
  )
  parser.add_argument(
      "--backbone",
      type=str,
      default="resnet50",
      choices=[
          "resnet18",
          "resnet34",
          "resnet50",
          "mobilenet_v2",
          "efficientnet_b0",
          "densenet121",
      ],
      help="Backbone architecture name",
  )
  parser.add_argument(
      "--train-img-dir",
      type=str,
      default="code/MonuSeg/MonuSeg/Training/TissueImages",
      help="Training tissue images directory",
  )
  parser.add_argument(
      "--train-mask-dir",
      type=str,
      default="code/MonuSeg/MonuSeg/Training/GroundTruth",
      help="Training ground truth masks directory",
  )
  parser.add_argument(
      "--test-img-dir",
      type=str,
      default="code/MonuSeg/MonuSeg/Test/TissueImages",
      help="Test tissue images directory",
  )
  parser.add_argument(
      "--test-mask-dir",
      type=str,
      default="code/MonuSeg/MonuSeg/Test/GroundTruth",
      help="Test ground truth masks directory",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default="output/panels",
      help="Directory to save generated comparison panels",
  )
  parser.add_argument(
      "--num-samples",
      type=int,
      default=5,
      help="Number of samples to visualize per set (-1 for all)",
  )
  parser.add_argument(
      "--target-size",
      type=int,
      nargs=2,
      default=[512, 512],
      help="Network input dimensions [Height, Width]",
  )
  parser.add_argument(
      "--display-size",
      type=int,
      nargs=2,
      default=[512, 512],
      help="Individual pane resolution [Height, Width]",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      default=0.5,
      help="Probability threshold for binarization",
  )
  parser.add_argument(
      "--title-tag",
      type=str,
      default="UNITPath Prediction",
      help="Header text on the prediction pane",
  )
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(args.output_dir, exist_ok=True)

  if not os.path.exists(args.checkpoint):
    print(f"Error: Model checkpoint not found at: {args.checkpoint}")
    return

  print(f"Loading checkpoint: {args.checkpoint} (Backbone: {args.backbone})")
  model = SegModel(
      backbone_name=args.backbone, pretrained=False, num_classes=1
  ).to(device)

  checkpoint = torch.load(args.checkpoint, map_location=device)
  state_dict = (
      checkpoint["model_state_dict"]
      if "model_state_dict" in checkpoint
      else (checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
  )
  model.load_state_dict(state_dict, strict=False)
  model.eval()

  ext = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

  # Gather train files
  if os.path.exists(args.train_img_dir):
    train_files = sorted(
        [f for f in os.listdir(args.train_img_dir) if f.lower().endswith(ext)]
    )
    if args.num_samples > 0:
      train_files = train_files[: args.num_samples]
  else:
    train_files = []

  # Gather test files
  if os.path.exists(args.test_img_dir):
    test_files = sorted(
        [f for f in os.listdir(args.test_img_dir) if f.lower().endswith(ext)]
    )
    if args.num_samples > 0:
      test_files = test_files[: args.num_samples]
  else:
    test_files = []

  if train_files:
    print(f"\nProcessing {len(train_files)} training samples...")
    process_set(
        model=model,
        img_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        img_list=train_files,
        source_label="TRAIN_SET",
        device=device,
        output_folder=args.output_dir,
        target_size=args.target_size,
        display_size=args.display_size,
        threshold=args.threshold,
        title_tag=args.title_tag,
    )

  if test_files:
    print(f"\nProcessing {len(test_files)} test samples...")
    process_set(
        model=model,
        img_dir=args.test_img_dir,
        mask_dir=args.test_mask_dir,
        img_list=test_files,
        source_label="TEST_SET",
        device=device,
        output_folder=args.output_dir,
        target_size=args.target_size,
        display_size=args.display_size,
        threshold=args.threshold,
        title_tag=args.title_tag,
    )

  print(f"\nCompleted! Generated panels saved to: {args.output_dir}")


if __name__ == "__main__":
  main()