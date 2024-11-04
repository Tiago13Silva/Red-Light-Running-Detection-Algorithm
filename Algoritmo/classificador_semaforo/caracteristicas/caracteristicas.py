from .hsv import HSV
from .cor import Cor
import cv2


class Caracteristicas:
    
    def __init__(self):
        self.avaliar_hsv = HSV()
        self.avaliar_cor = Cor()
        
        
    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        standard_im = cv2.resize(img.astype('uint8'), dsize=(32, 96))
        
        return standard_im
    
    
    def run(self, img):
        img = self.preprocess_image(img)
        
        _, _, sb, ss = self.avaliar_hsv.mask_image_get_vectors(img)

        brightness_dominant = self.avaliar_hsv.get_vectors_dominance(sb)
        saturation_dominant = self.avaliar_hsv.get_vectors_dominance(ss)
        color_dominant = self.avaliar_cor.get_color_dominance(img)
        
        # Print results
        print("Brightness color detected: ", brightness_dominant)
        print("Saturation color detected: ", saturation_dominant)
        print("Color detected: ", color_dominant)

        if (brightness_dominant != saturation_dominant and
            brightness_dominant != color_dominant and
            saturation_dominant != color_dominant):
            return -1
        elif brightness_dominant != saturation_dominant and saturation_dominant == color_dominant:
            return saturation_dominant
        
        return brightness_dominant