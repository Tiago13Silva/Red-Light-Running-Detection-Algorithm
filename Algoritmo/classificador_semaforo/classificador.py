from .caracteristicas.caracteristicas import Caracteristicas
from .cnn.rede_neuronal import RedeNeuronal
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

class Classificador:    
    
    def __init__(self, cnn=False):        
        if cnn:
            self._estados = ['GREEN', 'RED', 'YELLOW', 'OFF']
            self._classificador = RedeNeuronal()
        else:
            self._estados = ['RED', 'YELLOW', 'GREEN', 'OFF']
            self._classificador = Caracteristicas()
        
        
    def classificar(self, img):
        if img is None:
            print("Erro ao tentar ler imagem. Por favor forneça uma imagem válida!")
            return -1
        elif isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                print("Erro ao tentar ler imagem. Por favor forneça uma diretoria válida!")
                return -1
                    
        resultado = self._classificador.run(img)
        if resultado == -1:
            print("Não foi possível realizar uma classificação correta.")
        
        classificacao = self._estados[resultado]
        print("Cor detetada: ", classificacao)
        return classificacao
    
    
    def avaliar_recursos(self, img):
        max_cpu = 0
        max_ram = 0 
        start_time = time.time()
        
        self.classificar(img)
        
        # Get CPU usage percentage
        current_cpu = psutil.cpu_percent(interval=1)
        # Get RAM usage in MB
        current_ram = psutil.Process().memory_info().rss / (1024 * 1024)

        # Check and update the maximum values
        if current_cpu > max_cpu:
            max_cpu = current_cpu

        if current_ram > max_ram:
            max_ram = current_ram
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("Máximo CPU utilizado: ", max_cpu)
        print("Máximo RAM utilizado: ", max_ram)
        print("Tempo total passado: ", total_time)