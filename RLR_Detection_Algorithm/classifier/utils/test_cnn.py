from ..cnn.neural_network import NeuralNetwork
# import cv2
# import numpy as np
# from pathlib import Path
# import pickle

train_dir = 'classificador_semaforo/train'
test_dir = 'classificador_semaforo/test'

epochs = 100

cnn = NeuralNetwork()

cnn.train(train_dir, epochs)
cnn.save_model()
cnn.load_model('classificador_semaforo/cnn/model.pth')
cnn.test(test_dir)


# def load_regs_coords(filename):
#     if not Path(filename).exists():
#         raise FileNotFoundError("Source path does not exist.")
    
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
#         if 'violation_reg' in data and 'light_reg' in data:
#             violation_reg = data['violation_reg']
#             light_reg = data['light_reg']
#             print(f"Regions coordinates loaded from {filename}")
#         else:
#             print("Error: Invalid file format. The file does not contain the expected data.")
            
#     return violation_reg, light_reg


# coords_file="coords.pickle"
# violation_reg, light_reg=load_regs_coords(coords_file)

# img = cv2.imread('class_test.png')
# for light in light_reg:
                    
#     light_region = np.array(light["polygon"].exterior.coords, dtype=np.int32)
#     light_img = img[light_region[0][1]:light_region[2][1], light_region[0][0]:light_region[2][0]]

#     result = cnn.predict(light_img)
    
#     print(f"Traffic Light {light['id']} with color {result}")
    
#     cv2.imshow(f"Traffic Light {light['id']} with color {result}", light_img)
    
#     cv2.waitKey(0)
    
# cv2.destroyAllWindows()


# from PIL import Image
# import matplotlib.pyplot as plt

# # Open the image using PIL
# img = Image.open('class_test.png')

# for light in light_reg:
#     # Get the coordinates of the polygon and crop the region of interest
#     light_region = np.array(light["polygon"].exterior.coords, dtype=np.int32)
    
#     # Extract the coordinates for cropping
#     left = light_region[0][0]
#     upper = light_region[0][1]
#     right = light_region[2][0]
#     lower = light_region[2][1]
    
#     # Crop the region using PIL
#     light_img = img.crop((left, upper, right, lower))
    
#     # Pass the cropped region to the CNN for prediction
#     result = cnn.predict(light_img, False)
    
#     print(f"Traffic Light {light['id']} with color {result}")
    
#     # Display the cropped region using matplotlib
#     plt.imshow(light_img)
#     plt.title(f"Traffic Light {light['id']} with color {result}")
#     plt.show()