"""
Test the backend video analysis streaming endpoint directly
"""
import requests
import json

API_URL = "http://localhost:8000"
video_path = r"C:\Users\rajea\Documents\classroom\test video\test video - real classrom.mp4"

print("Testing backend video analysis stream...")
print(f"Video: {video_path}")
print(f"Endpoint: {API_URL}/api/analyze-video-path")
print("-" * 80)

response = requests.post(
    f"{API_URL}/api/analyze-video-path",
    data={"video_path": video_path, "frameSkip": 5},
    stream=True,
)

if response.status_code != 200:
    print(f"ERROR: {response.status_code} - {response.text}")
    exit(1)

print("✓ Connection successful. Streaming events...\n")

layout_received = False
frame_count = 0
face_count = 0

for line in response.iter_lines():
    if not line:
        continue
    
    try:
        event = json.loads(line)
        event_type = event.get("type")
        
        if event_type == "metadata":
            print(f"[METADATA] {event['totalFrames']} frames, {event['fps']:.1f} fps, {event['duration']:.1f}s")
        
        elif event_type == "layout":
            layout_received = True
            seats = event.get("seats", [])
            print(f"\n[LAYOUT] Detected {len(seats)} seats in {event['rows']} rows:")
            for seat in seats[:5]:  # Show first 5
                print(f"  - {seat['label']}: row {seat['row']}, bbox {seat['bbox']}")
            if len(seats) > 5:
                print(f"  ... and {len(seats) - 5} more")
            print()
        
        elif event_type == "frame":
            frame_count += 1
            faces = event.get("faces", [])
            face_count += len(faces)
            
            if frame_count == 1:
                # Print first frame's face data in detail
                print(f"[FRAME 1] {len(faces)} faces detected:")
                for i, face in enumerate(faces[:3]):  # Show first 3
                    print(f"  Face {i+1}:")
                    print(f"    gridCell: {face.get('gridCell')}")
                    print(f"    attention: {face.get('attentionScore')}")
                    print(f"    activity: {face.get('activity')}")
                    print(f"    emotion: {face.get('emotion')}")
                    print(f"    matchedName: {face.get('matchedName') or '(none)'}")
                    print(f"    thumbnail: {'Yes' if face.get('thumbnail') else 'No'}")
                print()
            
            elif frame_count % 10 == 0:
                # Progress update every 10 frames
                avg_faces = face_count / frame_count
                print(f"[PROGRESS] Frame {frame_count}, avg {avg_faces:.1f} faces/frame")
        
        elif event_type == "complete":
            print(f"\n[COMPLETE] {event['framesProcessed']} frames, {event['totalFaceDetections']} total detections")
            break
    
    except json.JSONDecodeError:
        print(f"Warning: Could not parse line: {line[:100]}")

print("\n" + "=" * 80)
print("SUMMARY:")
print(f"  Layout event received: {layout_received}")
print(f"  Total frames processed: {frame_count}")
print(f"  Total face detections: {face_count}")
print(f"  Avg faces per frame: {face_count / frame_count if frame_count > 0 else 0:.1f}")
print("=" * 80)
