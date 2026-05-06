import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO(r"C:\Users\siddh\Desktop\count_begs_ver1\counting_begs_v2\run\content\runs\detect\train3\weights\best.pt")

# Initialize DeepSORT
tracker = DeepSort(
    max_age=20,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.85
)

# Video input
video_path = r"C:\Users\siddh\Desktop\count_begs_ver1\begs.MP4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    frame = cv2.resize(frame,(640,640))
    results = model(frame, conf=0.85)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # DeepSORT format: [x, y, w, h], confidence, class
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                cls
            ))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        x1, y1, x2, y2 = int(l), int(t), int(w), int(h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("DeepSORT Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()