import argparse
from pathlib import Path
from threading import Thread
import cv2
import pickle
from shapely import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from region_colors import RegionColors


class SetRegions:
    """
    The SetRegions class is responsible for setting, managing and interacting with regions of interest (ROI) 
    within a video stream. It allows users to define, visualize, manipulate, and save/load regions 
    for violation detection and traffic light monitoring. The regions are paired, with each pair 
    consisting of a violation region and a corresponding traffic light region.
    """
    
    def __init__(self):
        self.opt = self.parse_opt()
        
        self.violation_reg = []
        self.light_reg = []
        
        self.video = self.opt.source
        
        self.yolo_model = None
        self.load_yolo_model(self.opt.weights, self.opt.device)
        
        self.model_classes = self.yolo_model.model.names
        
        self.current_region = None
        
        self.videocapture = None
        
        self.running = True
        
        
    def parse_opt(self):
        """
        Parse command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolov8s.pt", help="initial weights path")
        parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--source", type=str, required=True, help="video file path")
        parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
        parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

        return parser.parse_args()
    
    
    def load_yolo_model(self, weights, device):
        """
        Load the YOLO model with the specified weights and device (CPU or GPU).
        """
        self.yolo_model = YOLO(f"{weights}")
        self.yolo_model.to("cuda") if device == True else self.yolo_model.to("cpu")
        
        
    def add_region_pair(self, entry_line, direction, coords=None):
        """
        Add a pair of regions (violation and traffic light) to the respective lists.
        """
        new_id = len(self.violation_reg)
        
        # Get regions pair color
        enum_index = new_id % len(RegionColors)
        reg_color = list(RegionColors)[enum_index].value
        
        if new_id == 0:
            violation_polygon = Polygon([(650, 550), (800, 550), (800, 800), (650, 800)])
        else: # Use last set region coordinates for easier usage
            violation_polygon = self.violation_reg[-1]["polygon"]
        
        # Create the violation region
        violation_reg = {
            "id": new_id,
            "name": f"Violation Region {new_id}",
            "polygon": violation_polygon,
            "entry_line": entry_line,
            "counts": 0,
            "violations": 0,
            "car_count": 0,
            "dragging": False,
            "reg_color": reg_color,  # BGR Value
            "text_color": (0, 0, 0),  # Region Text Color
        }
        
        self.violation_reg.append(violation_reg)
        
        # Default traffic light coordinates to adjust
        if coords == None:
            light_polygon = Polygon([(100, 0), (400, 0), (400, 175), (100, 175)])
        else: # Use the detected traffic light coordinates
            points = coords.tolist()
            new_coords = [(points[0], points[1]), (points[2], points[1]), (points[2], points[3]), (points[0], points[3])]
            light_polygon = Polygon(new_coords)

        # Create the traffic light region
        light_reg = {
            "id": new_id,
            "name": f"Traffic Light Region {new_id}",
            "polygon": light_polygon,
            "color": "Off",
            "direction": direction,
            "dragging": False,
            "reg_color": reg_color,  # BGR Value
            "text_color": (255, 255, 255),  # Region Text Color
            "text_box_color": (0, 0, 0),  # Text Box Color
        }
        
        self.light_reg.append(light_reg)
        
        
    def save_regs_coords(self, filename):
        """
        Save the coordinates of the violation and traffic light regions to a file.
        """
        if '.' not in filename:
            filename = filename + ".pickle"
            
        with open(filename, 'wb') as f:
            pickle.dump({'violation_reg': self.violation_reg, 'light_reg': self.light_reg}, f)
        print(f"Regions coordinates saved to {filename}")
    
        
    def load_regs_coords(self, filename):
        """
        Load the coordinates of the regions from a file.
        """
        if '.' not in filename:
            filename = filename + ".pickle"
            
        if not Path(filename).exists():
            raise FileNotFoundError(f"Source path '{filename}' does not exist.")
        
        # Load the regions
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if 'violation_reg' in data and 'light_reg' in data:
                self.violation_reg = data['violation_reg']
                self.light_reg = data['light_reg']
                print(f"Regions coordinates loaded from {filename}")
            else:
                print("Error: Invalid file format. The file does not contain the expected data.")


    def handle_user_input(self):
        """
        Handle user input to control the addition, saving, and quitting of regions.
        """
        help_text = [
            "Commands:",
            "  'help': display this help message."
            "  'add': adds a region pair.",
            "  'save': saves the current regions coordinates to a file.",
            "  'exit': quits the program.",
        ]
        
        while self.running:
            user_input = input("Enter command (help, add, save, exit): ").lower()

            if user_input == "help":
                print("\n".join(help_text))

            elif user_input == "add":
                direction = input("Enter direction: ")
                self.add_region_pair(direction)
                print(f"Region pair added with direction: {direction}")
                
            elif user_input == "save":
                filename = input("Please specify the file to save the regions coordinates: ")
                self.save_regs_coords(filename)
                
            elif user_input in ["exit", "quit", "q"]:
                self.running = False  # Stop both threads
                self.videocapture.release()
                break
            
            
    def draw_region(self, frame, region, line_thickness, region_thickness):
        """
        Draw a region on the frame.
        """
        region_label = str(region["id"])
            
        region_color = region["reg_color"]
        region_text_color = region["text_color"]

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
            region_color,
            -1,
        )
        cv2.putText(
            frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
        )
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
            
            
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handles mouse events for region manipulation.

        Parameters:
            event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Additional flags passed by OpenCV.
            param: Additional parameters passed to the callback (not used in this function).

        Global Variables:
            current_region (dict): A dictionary representing the current selected region.

        Mouse Events:
            - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
            - MOUSEMOVE: Moves the selected region if dragging is active.
            - LBUTTONUP: Ends dragging for the selected region.

        Notes:
            - This function is intended to be used as a callback for OpenCV mouse events.
            - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

        Example:
            >>> cv2.setMouseCallback(window_name, mouse_callback)
        """
        sensitivity = 10  # How close the mouse must be to an edge to be considered for resizing
        offset = 5 # Limit for the corner not overlap others
        
        combined_reg_list = self.violation_reg + self.light_reg
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # First check for a click near any corners of any polygon
            found_corner = False
            for region in combined_reg_list:
                polygon = region['polygon']
                corners = polygon.exterior.coords[:-1]  # Get all corners excluding the closing point which is a repeat

                for idx, (cx, cy) in enumerate(corners):
                    if abs(cx - x) <= sensitivity and abs(cy - y) <= sensitivity:
                        self.current_region = region
                        self.current_region['dragging'] = 'resize'
                        self.current_region['corner_index'] = idx  # Store which corner is being dragged
                        found_corner = True
                        break
                if found_corner:
                    break

            # If no corner was found, check if the click is inside the polygon for moving
            if not found_corner:
                for region in combined_reg_list:
                    if region['polygon'].contains(Point(x, y)):
                        self.current_region = region
                        self.current_region['dragging'] = 'move'
                        self.current_region['offset_x'] = x
                        self.current_region['offset_y'] = y
                        break

        elif event == cv2.EVENT_MOUSEMOVE and self.current_region:
            if self.current_region.get('dragging') == 'resize' and 'corner_index' in self.current_region:
                idx = self.current_region['corner_index']
                coords = list(self.current_region["polygon"].exterior.coords[:-1])
                
                # Do not permit corner overlaping
                newx = x
                newy = y
                if idx == 0:
                    x0max = coords[2][0]
                    y0max = coords[2][1]
                    if x >= (x0max - offset):
                        newx = x0max - offset
                    if y >= (y0max - offset):
                        newy = y0max - offset
                elif idx == 1:
                    x1min = coords[3][0]
                    y1max = coords[3][1]
                    if x <= (x1min + offset):
                        newx = x1min + offset
                    if y >= (y1max - offset):
                        newy = y1max - offset
                elif idx == 2:
                    x2min = coords[0][0]
                    y2min = coords[0][1]
                    if x <= (x2min + offset):
                        newx = x2min + offset
                    if y <= (y2min + offset):
                        newy = y2min + offset
                elif idx == 3:
                    x3max = coords[1][0]
                    y3min = coords[1][1]
                    if x >= (x3max - offset):
                        newx = x3max - offset
                    if y <= (y3min + offset):
                        newy = y3min + offset
                
                coords[idx] = (newx, newy)  # Update only the selected corner
                new_polygon = Polygon(coords)
                self.current_region['polygon'] = new_polygon
                
            elif self.current_region.get('dragging') == 'move':
                dx = x - self.current_region['offset_x']
                dy = y - self.current_region['offset_y']
                new_polygon = Polygon(
                    [(px + dx, py + dy) for px, py in self.current_region["polygon"].exterior.coords[:-1]]
                )
                self.current_region['polygon'] = new_polygon
                self.current_region['offset_x'] = x
                self.current_region['offset_y'] = y

        elif event == cv2.EVENT_LBUTTONUP and self.current_region:
            # End dragging
            self.current_region['dragging'] = None
            self.current_region['corner_index'] = None
        
        
    def set_regions(self):
        """
        Main method for processing video frames and allowing region interaction.
        """
        vid_frame_count = 0
        thread_started = False

        # Check source path
        if not Path(self.video).exists():
            raise FileNotFoundError(f"Source path '{self.video}' does not exist.")

        # Video setup
        self.videocapture = cv2.VideoCapture(self.video)

        # Iterate over video frames
        while self.videocapture.isOpened():
            success, frame = self.videocapture.read()
            if not success:
                break
            vid_frame_count += 1
            
            # Check if user wants to load regions from a file
            if vid_frame_count == 5:
                load_reg = input("Do you want to load regions from a file? [yes|no]\n")
                if load_reg == "yes":
                    filename = input("Please specify the filename:\n")
                    self.load_regs_coords(filename)
            
            # Check if user wants to detect traffic lights regions using YOLO
            if vid_frame_count == 10:
                add = input("Do you want to add more traffic light regions with YOLO model detections? [yes|no]\n")
                if add == "yes":
                    # Extract the results
                    results = self.yolo_model(frame)
                    
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu()
                        clss = results[0].boxes.cls.cpu().tolist()
                                            
                        for box, cls in zip(boxes, clss):
                            if(cls == 9):
                                frame_copy = frame.copy()
                                annotator = Annotator(frame_copy, line_width=self.opt.line_thickness, example=str(self.model_classes))
                                
                                annotator.box_label(box, str(self.model_classes[cls]), color=colors(cls, True))
                                
                                cv2.imshow("Traffic Light Detection", frame_copy)
                                cv2.waitKey(100)
                                
                                user_input = input("Found a traffic light! Do you want to add it? [yes|no]\n")
                                if user_input == "yes":
                                    entry_line = input("From which line will the vehicles enter in the violation region? [NORTH|SOUTH|WEST|EAST]\n")
                                    direction = input("What direction should the vehicles follow? [NORTH|SOUTH|WEST|EAST]\n")
                                    if (entry_line == direction):
                                        print("The entry line should be different from the direction!")
                                        raise Exception("The entry line should be different from the direction!")
                                    self.add_region_pair(entry_line, direction, box)

                                cv2.destroyWindow("Traffic Light Detection")
            
            # Thread to control the addition, saving, and quitting of regions.
            if vid_frame_count == 15 and not thread_started:
                # thread start to ask if wanna add or save regions
                input_thread = Thread(target=self.handle_user_input)
                input_thread.start()
                                
            if vid_frame_count > 5:
                combined_reg_list = self.violation_reg + self.light_reg
                
                if len(combined_reg_list) > 0:
                    for region in combined_reg_list:
                        self.draw_region(frame, region, self.opt.line_thickness, self.opt.region_thickness)
                        
            if vid_frame_count == 1:
                cv2.namedWindow("Regions Set")
                cv2.setMouseCallback("Regions Set", self.mouse_callback)
            cv2.imshow("Regions Set", frame)
            
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

        input_thread.join()
        del vid_frame_count
        cv2.destroyAllWindows()


    def run(self):
        self.set_regions()
        
        
if __name__ == "__main__":
    reg = SetRegions()
    reg.run()