import argparse
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import pickle
import time

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
# from ultralytics.solutions import speed_estimation, distance_calculation

from classifier.classifier import Classifier
from light_states import LightStates



class Algorithm:
    """
    The Algorithm class represents the proposed algorithm for detecting Red Light Running violations based
    on video analysis and advanced artificial intelligence techniques.
    The algorithm is optimized for deployment on embedded systems and uses the YOLOv8 object detection model
    to identify and classify vehicles and the ByteTrack algorithm to track their motion, while using a 
    CNN-based or Features-based classifier to determine the state of traffic lights.
    """

    def __init__(self):
        self.opt = self.parse_opt()
        
        self.track_history = defaultdict(list)
        
        self.current_reg = None
        
        self.violation_reg = []
        self.light_reg = []

        # Load regions coordinates of violations and traffic lights ROIs
        self.load_regs_coords(self.opt.regions)
        
        self.video = self.opt.source
        self.fourcc = None
        self.fps = None
        self.frame_height = None
        self.frame_width = None

        # Load YOLOv8 model for object detection
        self.yolo_model = None
        self.load_yolo_model(self.opt.weights, self.opt.device)
        
        if self.opt.classes:
            self.classes = [2, 3, 5, 7]

        self.model_classes = {2:'car', 3:'motorcicle', 5:'truck', 7:'bus'}

        self.light_class = Classifier(self.opt.cnn)
                
        self.buffer_frames = None
        
        # Create a set() to not count the same car
        self.cars_ids = set()
        self.violation_ids = set()
        
        # Store info for each car in each region
        self.cars_info = defaultdict(lambda: defaultdict(dict)) # Adds if is not existing
        
        
    def parse_opt(self):
        """
        Parse command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolov8n.pt", help="yolo weights file path")
        parser.add_argument("--device", default="cpu", help="device to use, cuda or cpu")
        parser.add_argument("--source", type=str, required=True, help="video file path")
        parser.add_argument("--regions", type=str, required=True, help="regions coordinates file path")
        parser.add_argument("--cnn", action="store_true", help="traffic light color classification with cnn")
        parser.add_argument("--classes", action="store_true", help="use vehicle classes only")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
        parser.add_argument("--track-thickness", type=int, default=2, help="tracking line thickness")
        parser.add_argument("--region-thickness", type=int, default=4, help="region thickness")

        return parser.parse_args()
                
        
    def load_regs_coords(self, filename):
        """
        Load region coordinates from a file for violation and light detection.
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"Source path '{filename}' does not exist.")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if 'violation_reg' in data and 'light_reg' in data:
                self.violation_reg = data['violation_reg']
                self.light_reg = data['light_reg']
                print(f"Regions coordinates loaded from {filename}")
            else:
                print("Error: Invalid file format. The file does not contain the expected data.")
            
            
    def load_yolo_model(self, weights, device):
        """
        Load the YOLOv8 model for object detection on the specified device (CPU or CUDA).
        If the model is an engine format, it must use a CUDA device for its execution.
        """
        if weights.split('.')[-1] == 'engine':
            if self.opt.device == 'cpu':
                raise Exception("To use the optimized model it is necessary to use cuda device.")
            self.yolo_model = YOLO(f"{weights}", task='detect')
            
        else:    
            self.yolo_model = YOLO(f"{weights}")
            self.yolo_model.to("cuda") if device == "cuda" else self.yolo_model.to("cpu")
        
        
    def classify_traffic_light(self, light_img, id):
        """
        Classify the traffic light's color and store it in the corresponding light region.
        """
        light_state = self.light_class.classify(light_img)
        
        self.light_reg[id]['color'] = LightStates[light_state].name
        self.light_reg[id]['text_box_color'] = LightStates[light_state].value
        
        
    def save_violation_video(self, dir, frames, car_id, date):
        """
        Save a video of a detected violation with the provided frames.
        """
        if len(frames) == 0:
            print("No frames to save.")
            return
        
        # Define the codec and create VideoWriter object
        video_path = str(dir / f"violation_of_vehicle_{car_id}_on_{date}.mp4")
        print(video_path)
        video_writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (self.frame_width, self.frame_height))

        # Check if the VideoWriter object was successfully created
        if not video_writer.isOpened():
            print(f"Error: VideoWriter could not be opened with the path: {video_path}")
            return

        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved successfully at {video_path}")
    
        
    def draw_region(self, frame, region, line_thickness, region_thickness):
        """
        Draws the region's polygon and text label on the frame.
        """
        region_color = region["reg_color"]
        region_text_color = region["text_color"]
        
        if region in self.light_reg:
            region_label = str(region["color"])
            text_box_color = region['text_box_color']
        else:
            region_label = str(region["counts"])
            text_box_color = region_color

        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

        text_size, _ = cv2.getTextSize(
            region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
        )
        
        if region in self.violation_reg:
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
        elif region in self.light_reg:
            text_x = polygon_coords[0][0]
            text_y = polygon_coords[0][1] - 5

        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            text_box_color,
            -1,
        )
        cv2.putText(
            frame, f"{region_label}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
        )
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
    
    
    def get_polygon_lines(self, region):
        """
        Return the 4 sides of the region polygon as LineString objects.
        """
        polygon = region["polygon"]
        coords = list(polygon.exterior.coords)
        lines = {
            "NORTH": LineString([coords[0], coords[1]]),
            "EAST": LineString([coords[1], coords[2]]),
            "WEST": LineString([coords[3], coords[0]]),
            "SOUTH": LineString([coords[2], coords[3]])
        }
        return lines
    
    
    def check_line_intersect(self, lines, track):
        """
        Check if a vehicle track intersects any of the region's boundary lines.
        
        Returns:
        - The line intersected, if any (NORTH, EAST, WEST, SOUTH).
        - Returns None if none is intersected.
        """
        if len(track) > 1:
            track_line = LineString(track)
            for name, line in lines.items():
                if track_line.intersects(line):
                    return name
        return None
    
    
    def add_car_info(self, region_id, car_id, line_intersect, time=None):
        """
        Add or update car information when it enters or exits a region.
        """
        if car_id not in self.cars_info[region_id]:
            light_color = False
            
            if self.light_reg[region_id]["color"] == LightStates.RED.name:
                light_color = True
                print(f"Vehicle {car_id} entered region {region_id} with RED light!")
            
            self.cars_info[region_id][car_id] = {
                'id': car_id,
                'track': self.track_history[car_id],
                'region': region_id,
                'entry': line_intersect,
                'entry_time': time,
                'red_light': light_color, # Traffic Light state on vehicle entry
                'out': None
            }
        else:
            # Update the 'out' field if the car id exists
            self.cars_info[region_id][car_id]['out'] = line_intersect
        
        
    def process(self):
        """
        Main method for detecting RLR violations, using all the parts developed.
        Process the input video stream, classify traffic lights, detects and tracks vehicles, and processes RLR detection.
        """
        # Frame counter
        vid_frame_count = 0
        
        try:
            source = int(self.video) # If it ir an integer, assume it ir a camera index (like 0, 1, etc.)
        except ValueError:
            source = self.video # If not, assume it is a file path
            
        if isinstance(source, int):
            video_name = f"camera_{self.video}" # Name for output file if using a camera
        else:
            # Check source path
            if not Path(source).exists():
                raise FileNotFoundError(f"Source path '{self.video}' does not exist.")
            video_name = Path(self.video).stem # Extract video file name without extension
        
        # Video setup
        videocapture = cv2.VideoCapture(source) # Open video stream
        videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set video size
        videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set video size
        self.frame_width, self.frame_height = int(videocapture.get(3)), int(videocapture.get(4)) # Get video dimensions
        self.fps, self.fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v") # Get frames per second (fps) and set codec

        # Output setup
        save_dir = increment_path(Path("output") / "exp", self.opt.exist_ok) # Create output directory
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.opt.save_img: video_writer = cv2.VideoWriter(str(save_dir / f"{video_name}.mp4"), self.fourcc, self.fps,
                                       (self.frame_width, self.frame_height)) # Initialize video writer if saving frames

        buffer_duration = 10  # Video buffer duration in seconds to keep the video in buffer (for saving violation clips)
        buffer_size = buffer_duration * self.fps
        self.buffer_frames = deque(maxlen=buffer_size) # A circular buffer to store the last frames read
        
        # Init speed-estimation obj
        # line_pts = [(500, 450), (1100, 550)]
        # speed_obj = speed_estimation.SpeedEstimator()
        # speed_obj.set_args(reg_pts=line_pts,
        #                 names=names,
        #                 view_img=True)
        
        # # Init distance-calculation obj
        # dist_obj = distance_calculation.DistanceCalculation()
        # dist_obj.set_args(names=names, view_img=True)
        
        mean_inf_time = 0
        mean_total_inf_time = 0
        
        start_time = time.time()

        # Iterate over video frames
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                break
            
            # Track total inference time for performance measurement
            start_total_inf_time = time.time()
            
            vid_frame_count += 1
            self.buffer_frames.append(frame)
            
            # Read the video in each 3 frames
            # if vid_frame_count % 3 != 0:
            #     continue
            
            for light in self.light_reg:
                
                # Extract the region of the traffic light
                light_region = np.array(light["polygon"].exterior.coords, dtype=np.int32)
                light_img = frame[light_region[0][1]:light_region[2][1], light_region[0][0]:light_region[2][0]]
                
                # Classify the traffic light color
                self.classify_traffic_light(light_img, light["id"])
                
                if self.opt.view_img:
                    # show the traffic light separately
                    cv2.imshow(f"Traffic Light {light['id']}", light_img)
                        
            # Extract the results from YOLO detections and tracking
            if self.opt.classes:
                start_inf_time = time.time() # Control inference time
                results = self.yolo_model.track(frame, persist=True, classes=self.classes, tracker='bytetrack.yaml') #'botsort.yaml'
                end_inf_time = time.time()
            else:           
                start_inf_time = time.time()
                results = self.yolo_model.track(frame, persist=True, tracker='bytetrack.yaml') #'botsort.yaml'
                end_inf_time = time.time()
            
            # Save inference time to calculate mean inference time
            inf_time = end_inf_time - start_inf_time
            mean_inf_time += inf_time
            
            # frame = speed_obj.estimate_speed(frame, results)
            
            # im0 = dist_obj.start_process(im0, results)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                # To annotate the class and bounding box of the detected objetct
                if self.opt.view_img:
                    annotator = Annotator(frame, line_width=self.opt.line_thickness, example=str(self.model_classes))

                # Iterate for each object detected results
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    # Filter for only vehicle classes
                    if cls in [2, 3, 5, 7]:
                        if self.opt.view_img:
                            annotator.box_label(box, str(self.model_classes[cls] + " " + str(track_id)), color=colors(cls, True))
                        
                        # Get the center of the detected vehicle bounding box
                        bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                        track = self.track_history[track_id]  # Save the tracking points (vehicle trajectory)
                        track.append((float(bbox_center[0]), float(bbox_center[1])))
                        if len(track) > 100: # Limit the number of points in the track
                            track.pop(0)
                        
                        # Tracking Lines plot
                        if self.opt.view_img:
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=self.opt.track_thickness)
                        
                        # Convert box center to Point for geometry calculations usage
                        point = Point((bbox_center[0], bbox_center[1]))
                        
                        # Iterate each violation region
                        for region in self.violation_reg:
                            reg_id = region["id"]
                            
                            # Check if the vehicle center is inside the region
                            if region["polygon"].contains(point):
                                region["counts"] += 1

                                if track_id not in self.cars_ids:
                                    self.cars_ids.add(track_id)
                                    region["car_count"] += 1
                                    
                            # Check if vehicle entered the region with the intersection of the trajectory and the region boundary lines
                            lines = self.get_polygon_lines(region)
                            line_intersect = self.check_line_intersect(lines, track)
                            print(track_id, line_intersect)
                            
                            # If there isn't information about the vehicle it is a region entry
                            # and if it enters from the defined entry line, then we save the vehicle info
                            if track_id not in self.cars_info[reg_id] and line_intersect is not None and line_intersect == self.violation_reg[reg_id]["entry_line"]:
                                entry_time = time.localtime()
                                entry_time_str = time.strftime("day_%Y-%m-%d_hours_%H-%M-%S", entry_time)
                                print(f"Entry on: {entry_time_str}"  )
                                self.add_car_info(reg_id, track_id, line_intersect, entry_time_str)
                                
                            # If there already is information it is a region out
                            # then add the out information and process if there is violation
                            if track_id in self.cars_info[reg_id] and line_intersect is not None \
                            and self.cars_info[reg_id][track_id]["out"] == None \
                            and line_intersect != self.cars_info[reg_id][track_id]["entry"]:
                                self.add_car_info(reg_id, track_id, line_intersect)

                                if self.cars_info[reg_id][track_id]["red_light"] == True and self.light_reg[reg_id]["direction"] == self.cars_info[reg_id][track_id]["out"]:
                                    self.violation_ids.add(track_id)
                                    region["violations"] += 1
                                
                                    # Display Violation
                                    violation_region = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                                    violation_img = frame[violation_region[0][1]:violation_region[2][1], violation_region[0][0]:violation_region[2][0]]
                                    
                                    if self.opt.view_img:
                                        n_violations = region["violations"]
                                        cv2.putText(violation_img, f"No of violations: {n_violations}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        cv2.imshow("Violations", violation_img)
                                    
                                    self.save_violation_video(save_dir, self.buffer_frames, track_id, self.cars_info[reg_id][track_id]["entry_time"])

            # Draw regions (Polygons/Rectangles)
            if self.opt.view_img:
                combined_reg_list = self.violation_reg + self.light_reg
                for region in combined_reg_list:
                    self.draw_region(frame, region, self.opt.line_thickness, self.opt.region_thickness)

            # Put text of number of cars passed in region
            # n_cars = counting_regions[1]["car_count"]
            # cv2.putText(frame, f"No of cars that passed inside region {n_cars}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if self.opt.view_img:
                if vid_frame_count == 1:
                    cv2.namedWindow("Prototype8")
                cv2.imshow("Prototype8", frame)

            # Save the processed video with annotations
            if self.opt.save_img:
                video_writer.write(frame)

            # Reset region
            for region in self.violation_reg:
                region["counts"] = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # Process times for performance evaluation
            end_total_inf_time = time.time()    
            total_inf_time = end_total_inf_time - start_total_inf_time
            mean_total_inf_time += total_inf_time
        
        print(f"\n\nNº of frames: {vid_frame_count}")
        print(f"Mean Inference time: {mean_inf_time/vid_frame_count}")
        print(f"Mean Total Inference time: {mean_total_inf_time/vid_frame_count}")
        
        del vid_frame_count
        if self.opt.save_img: video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time} seconds")


if __name__ == "__main__":
    algorithm = Algorithm()
    algorithm.process()