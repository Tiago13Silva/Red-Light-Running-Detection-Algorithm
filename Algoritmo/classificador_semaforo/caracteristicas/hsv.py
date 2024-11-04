import cv2
import numpy as np
import matplotlib.pyplot as plt


class HSV:
    
    def __init__(self):
        pass
    
    def mask_image_get_vectors(self, rgb_image):
        """
        Tries to identify highlights within the traffic light's inner region and removes a vector with the
        brightness history from top to bottom

        rgb_image: An RGB image of a traffic light
        return: The history vector
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        brightness = hsv[:,:,2]
        summed_brightness = np.sum(brightness, axis=1)
        saturation = hsv[:,:,1]  # Extract the saturation channel
        summed_saturation = np.sum(saturation, axis=1)

        return (brightness, saturation, summed_brightness, summed_saturation)
        
        
    def display_image_vectors(self, img):
        img_bright, img_sat, sb, ss = self.mask_image_get_vectors(img)

        # Show details of example image
        f, (org, bright, sat, b, s) = plt.subplots(1, 5, figsize=(10,5))
        org.set_title("Original")
        bright.set_title("Brightness")
        sat.set_title("Saturation")
        b.set_title("Brightness vector")
        s.set_title("Saturation vector")
        org.imshow(img)
        bright.imshow(img_bright, cmap='gray')
        sat.imshow(img_sat, cmap='gray')
        b.barh(range(len(sb)), sb)
        b.invert_yaxis()
        s.barh(range(len(ss)), ss)
        s.invert_yaxis()
        plt.show()


    def get_vectors_dominance(self, vector):
        n = len(vector)
        third = n // 3
        mean = [
            np.mean(vector[:third]),         # Top third
            np.mean(vector[third:2*third]),  # Middle third
            np.mean(vector[2*third:])        # Bottom third
        ]
        
        return np.argmax(mean)