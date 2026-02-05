# ===================================================================
# STEP 8: VIDEO INFERENCE DEMO
# ===================================================================
# Run real-time object detection on any video file using your trained model.
# ===================================================================

import torch
from pathlib import Path
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
runs_dir = Path("runs")
runs_dir.mkdir(exist_ok=True)

print("="*70)
print("STEP 8: VIDEO INFERENCE DEMO")
print("="*70)

# ===================================================================
# PART 1: Load Trained Model
# ===================================================================

print("\n" + "-"*70)
print("PART 1: Loading Trained Model")
print("-"*70)


    

        
    
best_model_path = r"training_results\best_model.pt" 
trained_model = YOLO(str(best_model_path))
print(f"\n  ✓ Model loaded from: {best_model_path}")



# ===================================================================
# PART 2: Video Input Configuration
# ===================================================================

print("\n" + "-"*70)
print("PART 2: Video Input Configuration")
print("-"*70)

print("\n  Enter the path to your video file.")
print("  Supported formats: .mp4, .avi, .mov, .mkv, .wmv, etc.")
print("\n  Examples:")
print("    - C:/Videos/traffic_video.mp4")
print("    - ../test_video.mp4")
print("    - https://youtube.com/watch?v=xxxxx  (YouTube URL)")

video_path = input("\n  Video path (or URL): ").strip('"')

# ===================================================================
# PART 3: Inference Configuration
# ===================================================================

print("\n" + "-"*70)
print("PART 3: Inference Configuration")
print("-"*70)

print("\n  Configure inference parameters:")
print("    - conf: Confidence threshold (0-1, default: 0.25)")
print("    - iou: NMS IoU threshold (0-1, default: 0.45)")
print("    - save: Save output video (default: True)")
print("    - show: Display live preview (default: False)")

conf_thresh = input("\n  Confidence threshold [0.25]: ").strip()
conf_thresh = float(conf_thresh) if conf_thresh else 0.25

iou_thresh = input("  NMS IoU threshold [0.45]: ").strip()
iou_thresh = float(iou_thresh) if iou_thresh else 0.45

save_output = input("  Save output video? [Y/n]: ").strip().lower()
save_output = save_output != 'n'

show_live = input("  Show live preview? [y/N]: ").strip().lower()
show_live = show_live == 'y'

# ===================================================================
# PART 4: Run Video Inference
# ===================================================================

print("\n" + "-"*70)
print("PART 4: Running Video Inference")
print("-"*70)

print(f"\n  Processing video: {video_path}")
print(f"  Confidence: {conf_thresh}")
print(f"  IoU: {iou_thresh}")
print(f"  Save: {save_output}")
print(f"  Show preview: {show_live}")

print("\n  " + "="*66)
print("  STARTING VIDEO INFERENCE...")
print("  " + "="*66)

# Run inference on video
results = trained_model.predict(
    source=video_path,
    conf=conf_thresh,
    iou=iou_thresh,
    device=DEVICE,
    save=save_output,
    show=show_live,
    save_dir=runs_dir / "video_predictions",
    exist_ok=True,
    verbose=True,
    line_width=1,
)

print("\n  " + "="*66)
print("  VIDEO INFERENCE COMPLETE!")
print("  " + "="*66)

# ===================================================================
# PART 5: Output Summary
# ===================================================================

print("\n" + "-"*70)
print("PART 5: Output Summary")
print("-"*70)

# Get results
if isinstance(results, list):
    result = results[0] if results else None
else:
    result = results

print(f"\n  Input video:  {video_path}")

# Find the output video
output_dir = runs_dir / "video_predictions"
if output_dir.exists():
    output_videos = list(output_dir.rglob("*.mp4")) + list(output_dir.rglob("*.avi"))
    if output_videos:
        print(f"  Output video: {output_videos[-1]}")
        print(f"\n  Output saved to: {output_videos[-1].absolute()}")
    else:
        print(f"  Output directory: {output_dir.absolute()}")

print("\n" + "="*70)
print("✓ STEP 8: VIDEO INFERENCE COMPLETE!")
print("="*70)

print("\n  Tips:")
print("    - Use a lower confidence threshold (0.15-0.20) to detect more objects")
print("    - Use a higher confidence threshold (0.30-0.50) for fewer false positives")
print("    - Set show=True to see live predictions (slower processing)")
print("    - Output video will have the same name as input with predictions drawn")