import numpy as np


class Cor:
    def get_color_dominance(self, rgb_image):
        """This function searches for a very dominant red, yellow or green color within the traffic lights
        inner image region and independent of it's position
        
        rgb_image: The traffic light image in RGB
        return: A vector containing the percentage of red, yellow and green, (NOT RGB channels!) within the image
        """
        
        agg_colors = [0,0,0]
        
        threshold_min = 140
        threshold_min_b = 120
        threshold_rel = 0.75
        total_pixels = len(rgb_image)*len(rgb_image[1])

        for row_index in range(len(rgb_image)):
            cur_row = rgb_image[row_index]
            for col_index in range(len(rgb_image[0])):
                pixel = cur_row[col_index]
                if pixel[0]>threshold_min and pixel[1]<pixel[0]*threshold_rel and pixel[2]<pixel[0]*threshold_rel:
                    agg_colors[0] += 1
                if pixel[0]>threshold_min and pixel[1]>threshold_min and pixel[2]<pixel[0]*threshold_rel:
                    agg_colors[1] += 1
                if pixel[1]>threshold_min and pixel[0]<pixel[1]*threshold_rel and pixel[2]>threshold_min_b:
                    agg_colors[2] += 1

        agg_colors = np.array(agg_colors)/float(total_pixels)

        # Try to identify the image by dominant colors
        color_dominant = np.argmax(agg_colors)

        # Thresholds for dominant colors 
        # dominant_sure_threshold = 0.15
        dominant_threshold = 0.015

        if agg_colors[color_dominant]>dominant_threshold:
            return color_dominant
        else:
            return -1