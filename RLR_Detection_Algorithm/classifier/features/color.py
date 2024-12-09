import numpy as np


class Color:
    def get_color_dominance(self, rgb_image):
        """
        Analyzes an RGB image to determine if there is a dominant red, yellow, or green color.
        This is particularly useful for identifying the state of traffic lights within an image.

        Parameters:
        - rgb_image: A 3D NumPy array representing the traffic light image in RGB format.

        Returns:
        - A vector containing the percentage of red, yellow, and green within the image.
          Returns:
          - 0 for RED
          - 1 for YELLOW
          - 2 for GREEN
          - -1 for OFF (if no dominant color is identified)
        """
        # Initialize a vector to count pixels of dominant red, yellow, and green
        agg_colors = [0,0,0]
        
        # Define thresholds for identifying colors
        threshold_min = 140
        threshold_min_b = 120
        threshold_rel = 0.75
        total_pixels = len(rgb_image)*len(rgb_image[1])

        # Iterate through each pixel in the image
        for row_index in range(len(rgb_image)):
            cur_row = rgb_image[row_index]
            for col_index in range(len(rgb_image[0])):
                pixel = cur_row[col_index] # Pixel RGB values
                
                # Check if the pixel is predominantly red (high red)
                if pixel[0]>threshold_min and pixel[1]<pixel[0]*threshold_rel and pixel[2]<pixel[0]*threshold_rel:
                    agg_colors[0] += 1
                    
                # Check if the pixel is predominantly yellow (high red and green, low blue)
                if pixel[0]>threshold_min and pixel[1]>threshold_min and pixel[2]<pixel[0]*threshold_rel:
                    agg_colors[1] += 1
                    
                # Check if the pixel is predominantly green (high green with some blue)
                if pixel[1]>threshold_min and pixel[0]<pixel[1]*threshold_rel and pixel[2]>threshold_min_b:
                    agg_colors[2] += 1

        # Normalize the color counts by the total number of pixel
        agg_colors = np.array(agg_colors)/float(total_pixels)

        # Determine the most dominant color based on the highest percentage
        color_dominant = np.argmax(agg_colors)

        # Define a threshold to consider a color as dominant
        dominant_threshold = 0.015 # Minimum percentage required to classify a dominant color

        # Return the dominant color index if the percentage exceeds the threshold
        if agg_colors[color_dominant]>dominant_threshold:
            return color_dominant # 0=RED, 1=Yellow, 2=Green
        else:
            return -1 # OFF