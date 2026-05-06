import cv2
from ultralytics import YOLO

# Load YOLO model (you can use yolov8n.pt, yolov8s.pt, etc.)
model = YOLO(r"C:\Users\siddh\Desktop\count_begs_ver1\runs\detect\potato_bag_detection\yolov8m_run1\weights\best.pt")

# Input video path
video_path = r"C:\Users\siddh\Desktop\count_begs_ver1\begs.MP4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Run YOLO detection
    frame = cv2.resize(frame, (640, 640))
    results = model.track(frame, conf=0.5, imgsz=640, persist=True)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show frame (optional)
    cv2.imshow("YOLO Detection", annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()