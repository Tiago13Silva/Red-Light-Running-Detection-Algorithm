from pathlib import Path
import os
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from roboflow import Roboflow
import pandas as pd


# rf = Roboflow(api_key="SfARMQ6fU7402JWTZkHK")
# project = rf.workspace("amrsayed").project("traffic-light-1cygo-bvchu")
# version = project.version(1)
# dataset = version.download("tensorflow")

model = YOLO('yolov8s.pt')

current_region = None
counting_regions = [
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(1075, 246), (1093, 246), (1093, 282), (1075, 282)]),  #SeeJK1
        # "polygon": Polygon([(977, 234), (995, 234), (995, 281), (977, 281)]),  #SeeJK1
        # "polygon": Polygon([(676, 338), (700, 338), (700, 385), (676, 385)]), #Canmore1
        # "polygon": Polygon([(1165, 353), (1189, 353), (1189, 400), (1165, 400)]), #Canmore2
        # "polygon": Polygon([(1557, 1), (1581, 1), (1581, 47), (1557, 47)]), #Teste2
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]
img_counter = {
    "green": 129,
    "yellow": 168,
    "red": 196,
    "off": 60
}


def mouse_callback(event, x, y, flags, param):
    global current_region
    sensitivity = 10  # How close the mouse must be to an edge to be considered for resizing

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            polygon = region['polygon']
            min_x, min_y, max_x, max_y = polygon.bounds
            # Check if the click is near any of the edges of the rectangle
            if min_x - sensitivity <= x <= min_x + sensitivity or max_x - sensitivity <= x <= max_x + sensitivity or min_y - sensitivity <= y <= min_y + sensitivity or max_y - sensitivity <= y <= max_y + sensitivity:
                current_region = region
                current_region['dragging'] = 'resize'
                current_region['resize_edge'] = (
                    'left' if abs(x - min_x) < sensitivity else 'right' if abs(x - max_x) < sensitivity else None,
                    'top' if abs(y - min_y) < sensitivity else 'bottom' if abs(y - max_y) < sensitivity else None,
                )
            elif polygon.contains(Point((x, y))):
                # Click is inside the rectangle but not near the edges
                current_region = region
                current_region['dragging'] = 'move'
                current_region['offset_x'] = x
                current_region['offset_y'] = y

    elif event == cv2.EVENT_MOUSEMOVE and current_region is not None:
        dx = x - current_region.get('offset_x', x)
        dy = y - current_region.get('offset_y', y)

        if current_region['dragging'] == 'move':
            # Move the rectangle
            new_polygon = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords[:-1]]
            )
            print(new_polygon)
            current_region['polygon'] = new_polygon
            current_region['offset_x'] = x
            current_region['offset_y'] = y
        elif current_region['dragging'] == 'resize':
            # Resize the rectangle
            resize_edge = current_region['resize_edge']
            coords = list(current_region["polygon"].exterior.coords[:-1])
            min_x, min_y, max_x, max_y = current_region["polygon"].bounds

            if 'left' in resize_edge:
                min_x += dx
            if 'right' in resize_edge:
                max_x += dx
            if 'top' in resize_edge:
                min_y += dy
            if 'bottom' in resize_edge:
                max_y += dy

            # Ensure the rectangle does not invert
            min_x, max_x = min(min_x, max_x), max(min_x, max_x)
            min_y, max_y = min(min_y, max_y), max(min_y, max_y)

            new_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
            current_region['polygon'] = new_polygon

            current_region['offset_x'] = x
            current_region['offset_y'] = y

    elif event == cv2.EVENT_LBUTTONUP and current_region is not None:
        current_region['dragging'] = False
            
            
def save_img(img, path, name):
    print("save")
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Source path '{path}' does not exist.")
    
    filename = name + '.jpg'
    file_path = os.path.join(path, filename)
    cv2.imwrite(file_path, img)


def run(source, save_dir):
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    videocapture = cv2.VideoCapture(source)
    
    vid_frame_count = 0

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        vid_frame_count += 1
        original_frame = frame.copy()

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)

            cv2.polylines(frame, [polygon_coords], isClosed=True, color=(37, 255, 225), thickness=2)
        
        top_left = polygon_coords[0]
        bottom_right = polygon_coords[2]
        light_region = original_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cv2.imshow("", light_region)

        if vid_frame_count == 1:
            cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
            cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
        cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)
        
        key = cv2.waitKey(500) & 0xFF

        if key == ord("g"):
            img_counter["green"] += 1
            g=img_counter["green"]
            save_img(light_region, save_dir+"green", f"green{g}")
            
        if key == ord("y"):
            img_counter["yellow"] += 1
            y=img_counter["yellow"]
            save_img(light_region, save_dir+"yellow", f"yellow{y}")
            
        if key == ord("r"):
            img_counter["red"] += 1
            r=img_counter["red"]
            save_img(light_region, save_dir+"red", f"red{r}")
            
        if key == ord("o"):
            img_counter["off"] += 1
            o = img_counter["off"]
            save_img(light_region, save_dir+"off", f"off{o}")

        if key == ord("q"):
            break

    del vid_frame_count
    videocapture.release()
    cv2.destroyAllWindows()
    
    
# def process_images(source_dir, save_dir):
#     if not Path(source_dir).exists():
#         raise FileNotFoundError(f"Source path '{source_dir}' does not exist.")

#     # Iterate through images in the directory
#     image_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
#     for img_file in image_files:
#         img_path = os.path.join(source_dir, img_file)
#         frame = cv2.imread(img_path)
        
#         if frame is None:
#             print(f"Failed to load image: {img_file}")
#             continue
        
#         original_frame = frame.copy()
        
#         found_traffic_light = False
        
#         # YOLOv8 detection
#         results = model(frame)
#         if results[0].boxes is not None:
#             boxes = results[0].boxes.xyxy.cpu()
#             clss = results[0].boxes.cls.cpu().tolist()

#             # Process only if the class is 9 (usually traffic lights)   
#             for box, cls in zip(boxes, clss):
#                 if(cls == 9):
#                     found_traffic_light = True
                    
#                     # Extract traffic light region using detected coordinates
#                     points = box.tolist()
#                     print(points)
#                     # new_coords = [(points[0], points[1]), (points[2], points[1]), (points[2], points[3]), (points[0], points[3])]
#                     # light_polygon = Polygon(new_coords)
#                     light_region = original_frame[int(points[1]):int(points[3]), int(points[0]):int(points[2])]
#                     cv2.rectangle(frame, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), (0, 255, 0), 2)
#                     cv2.imshow("Detected Traffic Light", frame)

#                     # Wait for user to label the light
#                     key = cv2.waitKey(0) & 0xFF

#                     if key == ord("g"):
#                         img_counter["green"] += 1
#                         g = img_counter["green"]
#                         save_img(light_region, save_dir+"green", f"green{g}")
                    
#                     elif key == ord("y"):
#                         img_counter["yellow"] += 1
#                         y = img_counter["yellow"]
#                         save_img(light_region, save_dir+"yellow", f"yellow{y}")
                    
#                     elif key == ord("r"):
#                         img_counter["red"] += 1
#                         r = img_counter["red"]
#                         save_img(light_region, save_dir+"red", f"red{r}")
                        
#                     elif key == ord("o"):
#                         img_counter["off"] += 1
#                         o = img_counter["off"]
#                         save_img(light_region, save_dir+"off", f"off{o}")

#                     if key == ord("q"):
#                         break  # Quit processing
                    
#         if found_traffic_light:
#             cv2.destroyWindow("Detected Traffic Light")
#             found_traffic_light = False
        
#         cv2.imshow("Image", original_frame)
#         cv2.waitKey(1000)
#         cv2.destroyWindow("Image")

#     cv2.destroyAllWindows()


def process_images_from_csv(image_folder, csv_file, save_dir):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    count = 0

    # Loop through each row in the CSV
    for index, row in data.iterrows():
        # Extract information from the CSV
        filename = row['filename']
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Construct the full image path
        image_path = os.path.join(image_folder, filename)

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found.")
            continue

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        count += 1

        # Crop the traffic light region based on the coordinates from the CSV
        light_region = image[ymin:ymax, xmin:xmax]

        # Display the cropped region (optional)
        cv2.imshow('Cropped Traffic Light', light_region)
        
        # Wait for user to label the light
        key = cv2.waitKey(0) & 0xFF

        if key == ord("g"):
            img_counter["green"] += 1
            g = img_counter["green"]
            save_img(light_region, save_dir+"green", f"green{g}")
        
        elif key == ord("y"):
            img_counter["yellow"] += 1
            y = img_counter["yellow"]
            save_img(light_region, save_dir+"yellow", f"yellow{y}")
        
        elif key == ord("r"):
            img_counter["red"] += 1
            r = img_counter["red"]
            save_img(light_region, save_dir+"red", f"red{r}")
            
        elif key == ord("o"):
            img_counter["off"] += 1
            o = img_counter["off"]
            save_img(light_region, save_dir+"off", f"off{o}")

        if key == ord("q"):
            break  # Quit processing
        
        print(count)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # run('teste2_3.mp4', 'train/')
    # process_images('dataset1/train/images', 'train/')
    process_images_from_csv('Traffic-Light-1/valid', 'Traffic-Light-1/valid/_annotations.csv', 'test/')