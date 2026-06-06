"""
SmartClass AI — CLI Test Interface
Processes the test video, runs annotations, and records the session using the structured core engine.
"""
import os
import sys
import cv2

# Add backend directory to path if needed to find core package
sys.path.append(os.path.dirname(__file__))

from core.engine import ClassroomEngine, log

def run_cli_test():
    video_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "test video"))
    video_path = os.path.join(video_dir, "Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
    output_path = os.path.join(os.path.dirname(__file__), "annotated_output.mp4")
    
    if not os.path.exists(video_path):
        log.error(f"Test video not found at: {video_path}")
        return

    # Instantiate unified engine
    engine = ClassroomEngine(max_seats=6)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process every frame for granular annotation validation
    frame_generator = engine.process_video_file(video_path, frame_skip=1)
    
    metadata = next(frame_generator, None)
    
    faces_detected_in_video = False
    
    for event in frame_generator:
        if "error" in event:
            log.error(f"Error: {event['error']}")
            break
            
        if event.get("type") == "frame":
            frame_idx = event["frameIndex"]
            faces = event["faces"]
            
            # Read the original frame to overlay visual elements
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                continue
                
            if faces:
                faces_detected_in_video = True
                for face in faces:
                    sid = face["id"]
                    bbox = face["bbox"]
                    activity = face["activity"]
                    attn = face["attentionScore"]
                    
                    # Annotate frame details
                    x1, y1, x2, y2 = bbox
                    # Change box color based on attention tier
                    color = (0, 255, 0) if attn >= 75 else (0, 255, 255) if attn >= 55 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"Seat {sid} | {activity} | {attn:.0f}%"
                    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            out.write(frame)
            
        elif event.get("type") == "complete":
            log.info(f"Annotated video output recorded successfully.")
    
    cap.release()
    out.release()
    log.info(f"Annotated video saved to: {output_path}")
    
    if not faces_detected_in_video:
        log.error("CRITICAL: No faces were detected in the entire video.")

if __name__ == "__main__":
    run_cli_test()
