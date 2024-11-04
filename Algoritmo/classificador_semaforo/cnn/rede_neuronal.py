from .cnn import CNN
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools

class RedeNeuronal:
    def __init__(self):
        self.__modelo = CNN()
        
        
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues): # title="Matriz de confusão"
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(6.5, 6.5))
        plt.imshow(cm, interpolation='none', cmap=cmap)
        plt.title(title)
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.ylabel('Classe verdadeira')
        # plt.xlabel('Classe prevista')
        plt.show()
        
        
    def treinar(self, treino_dir, n_epocas=100):
        # Data transformations (same preprocessing as Keras)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load train and validation datasets
        batch_size = 32
        treino_data = ImageFolder(treino_dir, transform=transform)
        
        total_size = len(treino_data)
        train_size = int(0.8 * total_size)
        valid_size = total_size - train_size
        
        treino_dataset, valid_dataset = torch.utils.data.random_split(treino_data, [train_size, valid_size])
    
        treino_loader = DataLoader(treino_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss, optimizer
        criterio = nn.CrossEntropyLoss()
        otimizador = torch.optim.RMSprop(self.__modelo.parameters(), lr=0.001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.__modelo.to(device)

        # Lists to store loss and accuracy per epoch
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        # Training loop
        for epoca in range(n_epocas):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            self.__modelo.train()

            for images, labels in treino_loader:
                images, labels = images.to(device), labels.to(device)
                otimizador.zero_grad()
                outputs = self.__modelo(images)
                loss = criterio(outputs, labels)
                loss.backward()
                otimizador.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            avg_loss = running_loss / len(treino_loader)
            train_accuracy = correct_train / total_train

            # Validation loop
            val_loss, correct_val, total_val = 0.0, 0, 0
            self.__modelo.eval()

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.__modelo(images)
                    loss = criterio(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            avg_val_loss = val_loss / len(valid_loader)
            val_accuracy = correct_val / total_val

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoca+1}/{n_epocas}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Plot loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.grid(True)
        plt.axis([0, n_epocas, 0, 1])
        plt.legend()
        plt.show()

    def testar(self, teste_dir):
        # Data transformations (same preprocessing as Keras)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load test data
        batch_size = 32
        teste_data = ImageFolder(teste_dir, transform=transform)
        teste_loader = DataLoader(teste_data, batch_size=batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.__modelo.to(device)

        y_true, y_pred = [], []

        # Prediction on test data
        self.__modelo.eval()
        with torch.no_grad():
            for images, labels in teste_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.__modelo(images)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred)
        print(classification_report(y_true, y_pred))
        self.plot_confusion_matrix(cm, classes=teste_data.classes)
            
            
    def save_model(self):
        torch.save(self.__modelo.state_dict(), 'classificador_semaforo/cnn/model.pth')
            
            
    def load_model(self, model_dir):
        self.__modelo.load_state_dict(torch.load(model_dir))
        
        
    def prever(self, img, bgr=True):
        # See if image is in BGR format
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert the image to PIL format
            transforms.Resize((32, 32)),  # Resize to the size the network expects
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet mean and std
        ])
        
        # Apply the preprocessing transformations
        input_tensor = preprocess(img)

        # Add a batch dimension (required by the model)
        input_batch = input_tensor.unsqueeze(0)
        
        self.__modelo.eval()
        
        with torch.no_grad():
            resultado = self.__modelo(input_batch)
            
        print(f"Resultado: {resultado}")

        # Assuming the model outputs raw scores for each class, get the predicted class
        max_predict_value, max_predict_id = torch.max(resultado, 1)
        
        if max_predict_value < 0.80:
            return -1

        # print(f'Predicted class: {previsao.item()}')
        
        return max_predict_id.item()
    
    
    def run(self, img):
        
        self.load_model('classificador_semaforo/cnn/model.pth')
        
        return self.prever(img)