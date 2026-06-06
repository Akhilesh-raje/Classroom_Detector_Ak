import os
import shutil

root_dir = r"c:\Users\rajea\Documents\classroom"
models_dir = os.path.join(root_dir, "models")
unwanted_dir = os.path.join(root_dir, "unwanted")
backend_dir = os.path.join(root_dir, "backend")
backend_core = os.path.join(backend_dir, "core")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(unwanted_dir, exist_ok=True)

# Helper to safely move files
def safe_move(src, dst):
    if not os.path.exists(src):
        print(f"Skipping: {src} does not exist.")
        return
    # If destination is a directory and already exists, we will move files into it
    # If destination is a file and already exists, rename it to avoid overwriting
    if os.path.exists(dst) and os.path.isfile(dst):
        print(f"Destination {dst} already exists! Moving with name modification.")
        base, ext = os.path.splitext(dst)
        count = 1
        new_dst = f"{base}_{count}{ext}"
        while os.path.exists(new_dst):
            count += 1
            new_dst = f"{base}_{count}{ext}"
        dst = new_dst
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Moving: {src} -> {dst}")
    shutil.move(src, dst)

# Active models
safe_move(os.path.join(root_dir, "yolo11s.pt"), os.path.join(models_dir, "yolo11s.pt"))
safe_move(os.path.join(backend_dir, "face_landmarker.task"), os.path.join(models_dir, "face_landmarker.task"))

# Unused/Duplicate models
safe_move(os.path.join(root_dir, "yolov8n.pt"), os.path.join(unwanted_dir, "yolov8n.pt"))
safe_move(os.path.join(backend_dir, "yolo11s.pt"), os.path.join(unwanted_dir, "yolo11s.pt"))

# Duplicate videos
safe_move(os.path.join(root_dir, "public", "test_video.mp4"), os.path.join(unwanted_dir, "test_video.mp4"))

# Face API models in public
public_models = os.path.join(root_dir, "public", "models")
if os.path.exists(public_models):
    safe_move(public_models, os.path.join(unwanted_dir, "face-api-models"))

# Legacy python compatibility wrappers
safe_move(os.path.join(backend_dir, "database.py"), os.path.join(unwanted_dir, "database.py"))
safe_move(os.path.join(backend_dir, "video_processor.py"), os.path.join(unwanted_dir, "video_processor.py"))
safe_move(os.path.join(backend_dir, "yolo_face.py"), os.path.join(unwanted_dir, "yolo_face.py"))

# Face engine move to core
safe_move(os.path.join(backend_dir, "face_engine.py"), os.path.join(backend_core, "face_engine.py"))

print("File movements complete successfully!")
