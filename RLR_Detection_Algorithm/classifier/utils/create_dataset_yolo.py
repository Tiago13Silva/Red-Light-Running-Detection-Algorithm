from pathlib import Path
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO


model = YOLO('yolov8s.pt')

source = '2024-04-23 10-31-54.mp4'

if not Path(source).exists():
    raise FileNotFoundError(f"Source path '{source}' does not exist.")

videocapture = cv2.VideoCapture(source)

vid_frame_count = 0

while videocapture.isOpened():
    success, frame = videocapture.read()
    if not success:
        break
    vid_frame_count += 1
    
    results = model(frame)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()

        count = 0

        # Process only if the class is 9 (usually traffic lights)   
        for box, cls in zip(boxes, clss):
            if(cls == 9):
                count += 1
                print(count)
                points = box.tolist()
                print(points)
                new_coords = [(points[0], points[1]), (points[2], points[1]), (points[2], points[3]), (points[0], points[3])]
                light_polygon = Polygon(new_coords)
                print(light_polygon)
                label = f"Traffic Light ({count})"
                cv2.rectangle(frame, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(points[0]), int(points[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    if vid_frame_count == 1:
        cv2.namedWindow("Util")
    cv2.imshow("Util", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
videocapture.release()