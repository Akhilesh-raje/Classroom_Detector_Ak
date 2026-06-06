# -*- coding: utf-8 -*-
"""
SmartClass AI — Live Webcam Feed Test
Opens your local webcam (device 0), processes frames in real-time using the YOLO engine,
annotates the preview window, and logs student attention states to the terminal.

Run: python scratch/test_webcam.py
"""
import sys
import os
import io
import cv2

# Set stdout encoding for safe Windows console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add backend directory to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.insert(0, backend_path)
from core.engine import ClassroomEngine, log
import subprocess
import json

def list_pnp_cameras():
    """Query Windows PNP for camera friendly names."""
    try:
        cmd = ["powershell", "-Command", "Get-PnpDevice -FriendlyName '*camera*' -Status OK | Select-Object FriendlyName, InstanceId | ConvertTo-Json"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if not isinstance(data, list):
                data = [data]
            return [item["FriendlyName"] for item in data if "FriendlyName" in item]
    except Exception:
        pass
    return []

def run_webcam_test():
    # Initialize ClassroomEngine (set max_seats to 3 for standard home/webcam tests)
    engine = ClassroomEngine(max_seats=3)
    
    print("\n" + "="*60)
    print("SmartClass AI — Real-time Camera Feed Test")
    print("="*60)
    
    # Search system PNP first and print results
    print("Searching for connected cameras using system query...")
    pnp_cameras = list_pnp_cameras()
    if pnp_cameras:
        print("\nConnected Camera Devices found:")
        for name in pnp_cameras:
            print(f"  • {name}")
        print()
    else:
        print("No camera names returned from system query.\n")

    print("Opening webcam device...")
    print("Press 'q' in the preview window to exit.")
    print("="*60 + "\n")
    
    # Scan available cameras to automatically pick the external USB webcam
    log.info("Scanning for active DirectShow video inputs...")
    available = []
    for i in range(5):
        # cv2.CAP_DSHOW prevents long startup timeouts on Windows
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened():
            ret, frame = c.read()
            if ret:
                available.append(i)
            c.release()
            
    if not available:
        log.error("CRITICAL: No active camera device detected. Please verify your webcam is connected.")
        return
        
    log.info(f"Available camera indices detected: {available}")
    
    # Auto-select: prioritize external USB cameras (higher index) over default integrated (index 0)
    current_idx_ptr = len(available) - 1 if len(available) > 1 else 0
    
    try:
        while True:
            selected_idx = available[current_idx_ptr]
            log.info(f"Opening camera index {selected_idx} (4K request)...")
            
            # Open with cv2.CAP_DSHOW to unlock high resolutions (4K) on Windows
            cap = cv2.VideoCapture(selected_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                log.error(f"Failed to open selected camera index {selected_idx}. Trying next one.")
                current_idx_ptr = (current_idx_ptr + 1) % len(available)
                continue

            # Request 4K UHD resolution (3840x2160)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"Camera opened successfully: index {selected_idx} at {w}x{h} resolution.")

            # Create resizable window and set default display size (keeps full 4K frame for processing)
            window_name = "SmartClass AI - Live Camera Feed (Press Q to Quit)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_w = 1280
            display_h = int(display_w * h / w) if w > 0 else 720
            cv2.resizeWindow(window_name, display_w, display_h)

            switch_camera = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Could not read frame from webcam.")
                    break

                # Mirror frame horizontally for standard user preview feel
                frame = cv2.flip(frame, 1)

                # Process single frame through unified engine (disabling persistent tracking logic to adapt to shifting camera angles)
                faces = engine.process_single_frame(frame, w, h, persist=True)

                if faces:
                    for face in faces:
                        sid = face["id"]
                        bbox = face["bbox"]
                        activity = face["activity"]
                        attn = face["attentionScore"]
                        
                        x1, y1, x2, y2 = bbox
                        # Color coding: Green=Attentive, Yellow=Moderate/Neutral/Fidget, Red=Distracted/Drowsy/Phone
                        color = (0, 255, 0) if attn >= 75 else (0, 255, 255) if attn >= 55 else (0, 0, 255)
                        
                        # Draw box and core label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Seat {sid} | {activity.upper()} | {attn:.0f}%"
                        cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        
                        # Detailed MediaPipe analytics overlays
                        hp = face.get("headPose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
                        eye_op = face.get("eyeOpenness", 1.0)
                        mouth_op = face.get("mouthOpen", 0.0)
                        emo = face.get("emotion", "neutral")
                        phone = face.get("phoneDetected", False)
                        crop_used = face.get("cropUsed", "none")
                        
                        # Draw details panel below bbox
                        details_y = y2 + 15
                        pose_str = f"Yaw:{hp['yaw']:+.1f} Pitch:{hp['pitch']:+.1f} ({crop_used})"
                        cv2.putText(frame, pose_str, (x1, min(h - 10, details_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
                        
                        details_y += 15
                        metrics_str = f"Eye:{eye_op:.2f} Mouth:{mouth_op:.2f} [{emo}]"
                        cv2.putText(frame, metrics_str, (x1, min(h - 10, details_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
                        
                        # Draw alert header for phone usage
                        if phone:
                            cv2.rectangle(frame, (x1, y1 - 32), (x2, y1 - 5), (0, 0, 255), -1)
                            cv2.putText(frame, "!!! PHONE !!!", (x1 + 5, y1 - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                            
                        # Highlight seat mapping on screen
                        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, color, -1)
                else:
                    cv2.putText(frame, "No face detected in feed", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Show status/control instructions overlay on the screen bottom
                status_bar_y = h - 20 if h > 100 else 40
                info_text = f"CAM: {selected_idx} ({w}x{h}) | 'N' to Cycle | '0'-'9' to Select | 'Q' to Quit"
                cv2.putText(frame, info_text, (20, status_bar_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Show preview window
                cv2.imshow(window_name, frame)

                # Check for key press to terminate or cycle camera
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    log.info("Switching to the next camera index...")
                    current_idx_ptr = (current_idx_ptr + 1) % len(available)
                    switch_camera = True
                    break
                elif ord('0') <= key <= ord('9'):
                    desired = key - ord('0')
                    if desired in available:
                        log.info(f"Switching directly to camera index {desired}...")
                        current_idx_ptr = available.index(desired)
                        switch_camera = True
                        break
                    else:
                        log.warning(f"Index {desired} not available. Active indices: {available}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not switch_camera or key == ord('q'):
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        log.info("Camera released and all windows closed.")

if __name__ == "__main__":
    run_webcam_test()
