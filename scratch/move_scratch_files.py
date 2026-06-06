import os
import shutil

scratch_dir = r"c:\Users\rajea\Documents\classroom\scratch"
webcam_dir = os.path.join(scratch_dir, "webcam")
video_runs_dir = os.path.join(scratch_dir, "video_runs")
mp_tests_dir = os.path.join(scratch_dir, "mediapipe_tests")

os.makedirs(webcam_dir, exist_ok=True)
os.makedirs(video_runs_dir, exist_ok=True)
os.makedirs(mp_tests_dir, exist_ok=True)

# Helper to safely move files
def safe_move(src, dst):
    src_path = os.path.join(scratch_dir, src)
    dst_path = os.path.join(dst, src)
    if not os.path.exists(src_path):
        print(f"Skipping: {src_path} does not exist.")
        return
    if os.path.exists(dst_path):
        print(f"Destination {dst_path} already exists! Moving with name modification.")
        base, ext = os.path.splitext(dst_path)
        count = 1
        new_dst = f"{base}_{count}{ext}"
        while os.path.exists(new_dst):
            count += 1
            new_dst = f"{base}_{count}{ext}"
        dst_path = new_dst
    
    print(f"Moving scratch file: {src} -> {dst_path}")
    shutil.move(src_path, dst_path)

# Move Webcam files
safe_move("test_webcam.py", webcam_dir)
safe_move("play_realtime.py", webcam_dir)

# Move Video runs files
safe_move("run_analysis.py", video_runs_dir)
safe_move("test_pipeline.py", video_runs_dir)
safe_move("annotated_output.mp4", video_runs_dir)
safe_move("first_frame.jpg", video_runs_dir)
safe_move("middle_frame.jpg", video_runs_dir)

# Move MediaPipe test files
safe_move("test_mp.py", mp_tests_dir)
safe_move("test_mp_v2.py", mp_tests_dir)
safe_move("test_mp_v3.py", mp_tests_dir)

print("Scratch folder organized successfully!")
