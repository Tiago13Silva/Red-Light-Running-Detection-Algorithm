from .features.features import Features
from .cnn.neural_network import NeuralNetwork
import cv2
import time
import psutil

class Classifier:
    """
    Classifier for traffic light color classification using either a CNN or a Features-based approach.
    """
    
    def __init__(self, cnn=False):        
        if cnn:
            # State labels for the CNN model (must be equal to the train folders order)
            self._states = ['GREEN', 'RED', 'YELLOW', 'OFF']
            self._classifier = NeuralNetwork()
        else:
            # State labels for the Features-based approach (must be as the evaluated regions order)
            self._states = ['RED', 'YELLOW', 'GREEN', 'OFF']
            self._classifier = Features()
        
        
    def classify(self, img):
        """
        Classifies an image to predict the state of a traffic light.

        Parameters:
        - img: Input image (either as a file path or as a NumPy array).

        Returns:
        - The predicted state (e.g., RED, YELLOW, GREEN, OFF).
        - Returns -1 if the image is invalid or classification fails.
        """
        if img is None:
            print("Erro ao tentar ler imagem. Por favor forneça uma imagem válida!")
            return -1
        elif isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                print("Erro ao tentar ler imagem. Por favor forneça uma diretoria válida!")
                return -1

        # Run the classifier's `run` method to get the classification result
        result = self._classifier.run(img)
        if result == -1:
            print("Não foi possível realizar uma classificação correta.")
        
        # Map the result index to the corresponding state
        classification = self._states[result]
        print("Predicted Color: ", classification)
        return classification
    
    
    def evaluate_resources(self, img):
        """
        Evaluates system resource usage (CPU and RAM) and time taken during image classification.

        Parameters:
        - img: Input image (either as a file path or as a NumPy array).

        Prints:
        - Maximum CPU usage (%).
        - Maximum RAM usage (in MB).
        - Total time taken (in seconds).
        """
        max_cpu = 0
        max_ram = 0 
        start_time = time.time()
        
        # Perform the classification
        self.classify(img)
        
        # Measure CPU and RAM usage after classification
        current_cpu = psutil.cpu_percent(interval=1)
        current_ram = psutil.Process().memory_info().rss / (1024 * 1024)

        # Check and update the maximum values
        if current_cpu > max_cpu:
            max_cpu = current_cpu
            
        if current_ram > max_ram:
            max_ram = current_ram
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("Maximum CPU usage: ", max_cpu)
        print("Maximum RAM usage: ", max_ram)
        print("Total time taken: ", total_time)