import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO(r"C:\Users\siddh\Desktop\count_begs_ver1\runs\detect\potato_bag_detection\yolov8m_run1\weights\best.pt")

# Initialize DeepSORT (FIXED values)
tracker = DeepSort(
    max_age=25,
    n_init=2,
    max_cosine_distance=0.85
)

# Video input
video_path = r"C:\Users\siddh\Desktop\count_begs_ver1\begs.MP4"
cap = cv2.VideoCapture(video_path)

# Counting variables
count = 0
counted_ids = set()
line_x = 250  # adjust based on your video

# Store previous positions
track_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))

    # YOLO detection (FIXED)
    results = model(frame, conf=0.4)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                cls
            ))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw counting line
    cv2.line(frame, ( line_x, 0 ), (line_x,640), (0, 0, 255), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        x1, y1, x2, y2 = int(l), int(t), int(r), int(b)

        # Center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box + ID
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        # cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)

        # Previous position
        prev_cx = track_positions.get(track_id, None)

        # Check crossing (Right to Left)
        if prev_cx is not None:
            if prev_cx > line_x and cx <= line_x:
                if track_id not in counted_ids:
                    count += 1
                    counted_ids.add(track_id) 
                    
        # Reverse counting
        
        # Update position
        track_positions[track_id] = cx

    # Show count
    cv2.putText(frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("DeepSORT Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()