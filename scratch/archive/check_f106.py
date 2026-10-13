import cv2, os, sys
sys.path.insert(0, 'backend')
from core.structure import TopologyBuilder
from core.topology import ClassroomTopology
from core.detector import detect_faces_and_phones

builder = TopologyBuilder((1920, 1080))
tmap = builder.build_from_geometry([])
topo = ClassroomTopology.from_structure_map(tmap)

cap = cv2.VideoCapture(os.path.join('test video', 'test video - real classrom.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, 106)
ret, frame = cap.read()
cap.release()

dets, _, _, _ = detect_faces_and_phones(frame, min_conf=0.25, persist=True)
for d in dets:
    if d.get("id") in (2, 11, 40):
        fb = d["bbox"]
        cx = (fb[0] + fb[2]) / 2.0
        cy = (fb[1] + fb[3]) / 2.0
        pb = d.get("person_bbox", [0,0,0,0])
        print(f"Frame 106: tid={d.get('id')} head=({cx:.1f}, {cy:.1f}) pb_bot={pb[3]}")
