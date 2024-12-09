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

class NeuralNetwork:
    """
    A class to classify the color of traffic light images using a CNN.
    """
    
    def __init__(self):
        self.__model = CNN()
        
        
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
        plt.show()
        
        
    def train(self, train_dir, n_epochs=100):
        """
        Trains the CNN model using a dataset located in train_dir.
        Parameters:
        - train_dir: Path to the training dataset directory.
        - n_epochs: Number of epochs for training.
        """
        # Data preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and split the dataset into training and validation subsets
        batch_size = 32
        train_data = ImageFolder(train_dir, transform=transform)
        
        total_size = len(train_data)
        train_size = int(0.8 * total_size)
        valid_size = total_size - train_size
        
        train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [train_size, valid_size])
    
        # Data loaders for batch processing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Define the loss function and optimizer
        criterio = nn.CrossEntropyLoss()
        otimizador = torch.optim.RMSprop(self.__model.parameters(), lr=0.001)

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.__model.to(device)

        # Initialize lists to track losses and accuracies
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        # Training loop
        for epoch in range(n_epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            self.__model.train() # Set model to training mode

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                otimizador.zero_grad() # Zero the gradient
                outputs = self.__model(images) # Forward pass
                loss = criterio(outputs, labels) # Calculate loss
                loss.backward() # Backpropagation
                otimizador.step() # Update model weights
                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            avg_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            # Validation loop
            val_loss, correct_val, total_val = 0.0, 0, 0
            self.__model.eval()  # Set model to evaluation mode

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.__model(images)
                    loss = criterio(outputs, labels)
                    val_loss += loss.item()

                    # Calculate validation accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            avg_val_loss = val_loss / len(valid_loader)
            val_accuracy = correct_val / total_val

            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Plot loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.grid(True)
        plt.axis([0, n_epochs, 0, 1])
        plt.legend()
        plt.show()


    def test(self, test_dir):
        """
        Evaluates the trained CNN model on a test dataset.
        Parameters:
        - test_dir: Path to the test dataset directory.
        """
        # Data preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the test dataset
        batch_size = 32
        test_data = ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.__model.to(device)

        # Lists to store true and predicted labels
        y_true, y_pred = [], []

        # Prediction on test data
        self.__model.eval() # Set the model to evaluation mode
        with torch.no_grad(): # Disable gradient computation for evaluation
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.__model(images)
                _, predicted = torch.max(outputs, 1) # Get the index of the max log-probability
                y_pred.extend(predicted.cpu().numpy()) # Store predictions
                y_true.extend(labels.cpu().numpy()) # Store true labels

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred)
        print(classification_report(y_true, y_pred))
        self.plot_confusion_matrix(cm, classes=test_data.classes)
            
            
    def save_model(self):
        """
        Saves the trained model to a file.
        """
        torch.save(self.__model.state_dict(), 'classifier/cnn/model.pth')
            
            
    def load_model(self, model_dir):
        """
        Loads the model from a file.
        Parameters:
        - model_dir: Path to the model file.
        """
        self.__model.load_state_dict(torch.load(model_dir))
        
        
    def predict(self, img, bgr=True):
        """
        Predicts the class of a given image.
        Parameters:
        - img: Input image in NumPy array format.
        - bgr: Boolean indicating if the image is in BGR format (default is True).
        Returns:
        - Predicted class index or -1 if confidence is below threshold.
        """
        # Convert BGR image to RGB if necessary
        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Data preprocessing transformations
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert the image to PIL format
            transforms.Resize((32, 32)),  # Resize image to 32x32 pixels
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet mean and std
        ])
        
        # Apply the preprocessing transformations
        input_tensor = preprocess(img)

        # Add a batch dimension (required by the model)
        input_batch = input_tensor.unsqueeze(0)
        
        self.__model.eval()
        
        # Disable gradient computation
        with torch.no_grad(): 
            result = self.__model(input_batch)
            
        print(f"result: {result}")

        # Get the predicted class and its confidence
        max_predict_value, max_predict_id = torch.max(result, 1)
        
        # Return -1 if confidence is below 80%
        if max_predict_value < 0.80:
            return -1
        
        # Return predicted class index
        return max_predict_id.item()
    
    
    def run(self, img):
        
        self.load_model('classifier/cnn/model.pth')
        
        return self.predict(img)