from .hsv import HSV
from .color import Color
import cv2


class Features:
    """
    A class to extract and evaluate features from traffic light images using HSV and color analysis.
    """
    
    def __init__(self):
        self.evaluate_hsv = HSV() # Instance of HSV class for brightness and saturation evaluation
        self.evaluate_color = Color() # Instance of Color class for color dominance evaluation
        
        
    def preprocess_image(self, img):
        """
        Preprocesses an input image for feature extraction:
        - Converts the image from BGR to RGB color space.
        - Resizes the image to a standard size.

        Parameters:
        - img: Input image in BGR format.

        Returns:
        - The preprocessed image in RGB format with dimensions (32, 96).
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 32x96 dimensions
        standard_im = cv2.resize(img.astype('uint8'), dsize=(32, 96))
        
        return standard_im
    
    
    def run(self, img):
        # Preprocess the input image
        img = self.preprocess_image(img)
        
        # Get brightness and saturation vectors using the HSV class
        _, _, sb, ss = self.evaluate_hsv.mask_image_get_vectors(img)

        # Determine dominant brightness, saturation and color regions
        brightness_dominant = self.evaluate_hsv.get_vectors_dominance(sb)
        saturation_dominant = self.evaluate_hsv.get_vectors_dominance(ss)
        color_dominant = self.evaluate_color.get_color_dominance(img)
        
        print("Brightness color detected: ", brightness_dominant)
        print("Saturation color detected: ", saturation_dominant)
        print("Color detected: ", color_dominant)

        # If all features disagree, return -1 (no consensus)
        if (brightness_dominant != saturation_dominant and
            brightness_dominant != color_dominant and
            saturation_dominant != color_dominant):
            return -1
        
        # If brightness disagrees but saturation and color agree, return the agreed dominant
        elif brightness_dominant != saturation_dominant and saturation_dominant == color_dominant:
            return saturation_dominant
        
        # Otherwise, return the agreed dominant
        return brightness_dominant