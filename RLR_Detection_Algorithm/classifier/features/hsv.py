import cv2
import numpy as np
import matplotlib.pyplot as plt


class HSV:
    def mask_image_get_vectors(self, rgb_image):
        """
        Analyzes the brightness and saturation channels of an input RGB image.
        Extracts the brightness and saturation channels and computes their summed values along the vertical axis.
        
        Parameters:
        - rgb_image: An RGB image of a traffic light.

        Returns:
        - brightness: Brightness channel (V) of the HSV image.
        - saturation: Saturation channel (S) of the HSV image.
        - summed_brightness: Sum of brightness values along each row.
        - summed_saturation: Sum of saturation values along each row.
        """
        # Convert RGB image to HSV color space
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Extract the brightness (V) channel
        brightness = hsv[:,:,2]
        
        # Sum brightness values along each row
        summed_brightness = np.sum(brightness, axis=1)
        
        # Extract the saturation (S) channel
        saturation = hsv[:,:,1]
        
        # Sum saturation values along each row
        summed_saturation = np.sum(saturation, axis=1)

        return (brightness, saturation, summed_brightness, summed_saturation)
        
        
    def display_image_vectors(self, img):
        """
        Visualizes the brightness and saturation channels and their respective vectors for the input image.
        
        Parameters:
        - img: An RGB image to analyze and visualize.
        """
        img_bright, img_sat, sb, ss = self.mask_image_get_vectors(img)

        f, (org, bright, sat, b, s) = plt.subplots(1, 5, figsize=(10,5))
        org.set_title("Original")
        bright.set_title("Brightness")
        sat.set_title("Saturation")
        b.set_title("Brightness vector")
        s.set_title("Saturation vector")
        
        # Plot original image
        org.imshow(img)
        
        # Display brightness and saturation as grayscale images
        bright.imshow(img_bright, cmap='gray')
        sat.imshow(img_sat, cmap='gray')
        
        # Plot brightness vector as a horizontal bar chart
        b.barh(range(len(sb)), sb)
        b.invert_yaxis()
        
        # Plot saturation vector as a horizontal bar chart
        s.barh(range(len(ss)), ss)
        s.invert_yaxis()
        
        plt.show()


    def get_vectors_dominance(self, vector):
        """
        Determines which third of the vector (color region) has the highest average value.
        
        Parameters:
        - vector: A 1D numerical vector (e.g., summed brightness or saturation values).
        
        Returns:
        - The index of the third (0 for top or RED, 1 for middle or YELLOW, 2 for bottom or GREEN) with the highest average value.
        """
        n = len(vector)
        
        # Determine the size of one-third of the vector
        third = n // 3
        
        mean = [
            np.mean(vector[:third]),         # Top third
            np.mean(vector[third:2*third]),  # Middle third
            np.mean(vector[2*third:])        # Bottom third
        ]
        
        # Return the index of the third with the highest average
        return np.argmax(mean)