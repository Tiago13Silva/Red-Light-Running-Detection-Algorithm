import cv2
import numpy as np
import matplotlib.pyplot as plt


TAMANHO_IMG = 32

def preprocess_image(img_name):
    img = cv2.imread(img_name)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    standard_im = cv2.resize(img.astype('uint8'), dsize=(32, 96))
    
    return standard_im


def mask_image_get_vectors(rgb_image):
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

    return (brightness, hsv[:,:,1], summed_brightness, summed_saturation)
    
    
def display_image_vectors(img):
    img_bright, img_sat, sb, ss = mask_image_get_vectors(img)

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


def get_vectors_dominance(vector):
    n = len(vector)
    third = n // 3
    mean = [
        np.mean(vector[:third]),         # Top third
        np.mean(vector[third:2*third]),  # Middle third
        np.mean(vector[2*third:])        # Bottom third
    ]
    
    return np.argmax(mean)


def get_color_dominance(rgb_image):
    """This function searches for a very dominant red, yellow or green color within the traffic lights
    inner image region and independent of it's position
    
    rgb_image: The traffic light image
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
    
    return agg_colors


tl_states = ['red', 'yellow', 'green']

img_name = "classificador_semaforo/train/green/green187.jpg"

img = preprocess_image(img_name)

display_image_vectors(img)

img_bright, img_sat, sb, ss = mask_image_get_vectors(img)

brightness_dominant = get_vectors_dominance(sb)
saturation_dominant = get_vectors_dominance(ss)

# Print results
print("Brightness color detected:", tl_states[brightness_dominant])
print("Saturation color detected:", tl_states[saturation_dominant])

agg_colors = get_color_dominance(img)

# Try to identify the image by dominant colors
color_dominant = np.argmax(agg_colors)

# Thresholds for dominant colors 
dominant_sure_threshold = 0.15
dominant_threshold = 0.015

if agg_colors[color_dominant]>dominant_threshold:
    print("By dominance detected color: {} ({})".format(tl_states[color_dominant], agg_colors))
else:
    print("No dominant color detected")
   
    
def update_color(color_idx):
    last = color_idx
    
    
def color_check(current_color_idx, last_color_idx):
    if last_color_idx is None or current_color_idx == (last_color_idx + 1) % 3:
        update_color(current_color_idx)
        return True
    return False