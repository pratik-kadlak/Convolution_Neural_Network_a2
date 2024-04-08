import numpy as np
import wandb
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
import argparse

wandb.login()
wandb.init(project="DL_Assignment_2")

def read_images(path):
    """
    Read images from a folder using PyTorch's ImageFolder and DataLoader,
    apply transformations, and return NumPy arrays for images and labels.

    Args:
    - path (str): Path to the folder containing images.

    Returns:
    - X (np.array): NumPy array of images with shape (num_images, channels, height, width).
    - y (np.array): NumPy array of labels with shape (num_images,).
    """
    data_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    dataset = ImageFolder(path, transform=data_transform)
        
    data = DataLoader(dataset, batch_size=32) 
    
    X = [] 
    y = []
    
    for image, label in tqdm(data):
        X.append(image) 
        y.append(label) 
        
    # Concatenate the lists of arrays along the batch dimension (axis=0)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
        
    return X, y


def shuffle_data(X, y):  
    """
    Shuffle data samples and their corresponding labels.

    Parameters:
    - X (numpy.ndarray): NumPy array containing data samples.
    - y (numpy.ndarray): NumPy array containing corresponding labels.

    Returns:
    - X_shuffled (numpy.ndarray): NumPy array containing shuffled data samples.
    - y_shuffled (numpy.ndarray): NumPy array containing corresponding shuffled labels.
    """
    
    # Combine X, y into a list of tuples
    data = list(zip(X, y))

    # Shuffle the combined data
    random.shuffle(data)

    # Unpack the shuffled data back into separate arrays
    X_shuffled, y_shuffled = zip(*data)

    # Convert the shuffled lists to NumPy arrays 
    X_shuffled = np.array(X_shuffled)
    y_shuffled = np.array(y_shuffled)
    
    return X_shuffled, y_shuffled


def create_dataloader(X, y, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader from input data and labels.

    Parameters:
    - X (numpy.ndarray): Input data array.
    - y (numpy.ndarray): Labels array.
    - batch_size (int, optional): Batch size for DataLoader (default=32).
    - shuffle (bool, optional): Whether to shuffle the data (default=True).

    Returns:
    - DataLoader: PyTorch DataLoader for the input data and labels.
    """
    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Create a TensorDataset from X_train_tensor and y_train_tensor
    dataset = TensorDataset(X_tensor, y_tensor)

    # Define batch size and create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


# # setting the path to dataset
# train_path = "/Users/pratikkadlak/Pratik/DeepLearning/DL_Assignment_2/inaturalist_12K/train"
# test_path = "/Users/pratikkadlak/Pratik/DeepLearning/DL_Assignment_2/inaturalist_12K/val"

# # reading the images 
# X_train, y_train = read_images(train_path)
# X_test, y_test = read_images(test_path)

# # shuffling the data
# X_train, y_train = shuffle_data(X_train, y_train)

# # making data loaders
# train_loader = create_dataloader(X_train, y_train, 32)
# test_loader = create_dataloader(X_test, y_test, 32)


# # Question 2

# ## GoogleNet

"""
Fine-tune a pre-trained GoogLeNet model on a custom dataset.

This code block loads a pre-trained GoogLeNet model, modifies the final fully connected (FC) layer
to have 10 output classes, defines a loss function (CrossEntropyLoss) and optimizer (SGD), and trains
the model on a custom dataset using the specified number of epochs.

The training loop iterates through each epoch, performing forward and backward passes, and evaluates
the model's accuracy on a test dataset after each epoch. The trained model is saved to a file named
'googlenet_model.pth' after training.

"""

def train_googlenet():
    # Define GoogLeNet model
    model = models.googlenet(pretrained=True)  # Load pre-trained weights
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify final FC layer

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Access GPU if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)


    # Train the model
    num_epochs = 10  # Number of training epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for images, labels in train_loader:  # Iterate over batches of training data
            images, labels = images.to(device), labels.to(device)  # Move data to device (CPU or GPU)
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(images)  # Forward pass: compute predicted outputs
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update model parameters based on gradients

        # Evaluate the model after each epoch
        model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization)
        correct = 0  # Initialize number of correctly predicted samples
        total = 0  # Initialize total number of samples
        with torch.no_grad():  # Disable gradient tracking for evaluation
            for images, labels in test_loader:  # Iterate over batches of test data
                images, labels = images.to(device), labels.to(device)  # Move data to device
                outputs = model(images)  # Forward pass: compute predicted outputs
                _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
                total += labels.size(0)  # Update total count of samples
                correct += (predicted == labels).sum().item()  # Count correct predictions

        accuracy = 100 * correct / total  # Calculate accuracy percentage
        print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%')  # Print test accuracy after each epoch

    # Save the trained model
    torch.save(model.state_dict(), 'googlenet_model.pth')


# ## 1. Freezing all layers except the last layer:
# - Freeze all layers except the final classification layer.
# - Fine-tune only the weights of the last layer during training.

# In[ ]:

"""
Train a model using a pre-trained GoogLeNet architecture for a specific classification task.

1. Load the pre-trained GoogLeNet model and modify the last layer for the desired number of output classes.
2. Freeze all layers except the last layer to only train the new classifier layer.
3. Define the optimizer (Adam) and loss function (CrossEntropyLoss).
4. Train the model using the specified data loader for a certain number of epochs.

Args:
- train_loader (torch.utils.data.DataLoader): DataLoader for training data.
- num_epochs (int): Number of training epochs (default is 10).
"""

# Train the model
def freeze_all(num_epochs=10):
    # num_classes is the number of classes in your dataset
    num_classes = 10

    # Load pre-trained GoogLeNet
    model = models.googlenet(pretrained=True)

    # Freeze all layers except the last layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last layer for your specific classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) 

    # Access GPU if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(train_loader):
            images , labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')


def test_model(model, criterion, test_loader):
    """
    Test a neural network model using the specified criterion and data loader.

    Args:
    - model (torch.nn.Module): The trained neural network model to evaluate.
    - criterion (torch.nn.Module): The loss function used for evaluation.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
    - None
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):  # assuming you have a DataLoader for test data
            images , labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


# ## 2. Fine-tuning up to a certain number of layers:
# 
# - Freeze the initial layers (e.g., convolutional layers) and fine-tune only the later layers (e.g., fully connected layers).
# - Experiment with different values of 'k' to find the optimal number of layers to fine-tune.


# Train the model
def freeze_k_layers(num_epochs=10):
    # Load pre-trained GoogLeNet
    model = models.googlenet(pretrained=True)

    # Define the number of layers to fine-tune (k)
    k = 5  # Example: Fine-tune the last 5 layers

    # Freeze layers up to k
    if k > 0:
        for i, child in enumerate(model.children()):
            if i < k:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                break

    # Modify the classifier for your specific classification task
    num_ftrs = model.fc.in_features
    num_classes = 10  # Change this to your actual number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move the model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')



def test_model(model, criterion, test_loader):
    """
    Evaluate a trained neural network model using the specified criterion and data loader for test data.

    Args:
    - model (torch.nn.Module): The trained neural network model to evaluate.
    - criterion (torch.nn.Module): The loss function used for evaluation.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
    - None
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):  # assuming you have a DataLoader for test data
            images , labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


# ## 3. Feature extraction using pre-trained models:
# 
# - Use pre-trained models like GoogLeNet, InceptionV3, ResNet50, etc., as feature extractors.
# - Remove the final classification layer and use the extracted features as inputs to a smaller model (e.g., a simple feedforward neural network).
# - Train the smaller model on the extracted features to classify images.


# Define a smaller feedforward neural network for classification
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def feature_extraction_model():
    # Set device (GPU if available, otherwise CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load pre-trained GoogLeNet without the final classification layer
    googlenet = models.googlenet(pretrained=True).to(device)
    googlenet = nn.Sequential(*list(googlenet.children())[:-1])  # Remove the final layer

    # Extract features using GoogLeNet
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            features = googlenet(images).squeeze()  # Remove the batch dimension
            features_list.append(features)
            labels_list.append(labels)

    # Concatenate features and labels
    features = torch.cat(features_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)

    # Define the input size for the classifier based on the extracted features
    input_size = features.size(1)

    # Initialize the simple classifier and move it to the device
    classifier = SimpleClassifier(input_size, hidden_size=128, num_classes=10).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Train the classifier on the extracted features
    num_epochs = 10
    for epoch in range(num_epochs):
        classifier.train()  # Set the model to training mode
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0) * 100

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy:.2f}%')

    # if you want to test the model on test data
    test_feature_extraction_model(classifier=classifier, googlenet=googlenet)


def test_feature_extraction_model(classifier, googlenet):
    # Initialize lists to store predicted labels and ground truth labels
    predicted_labels = []
    true_labels = []

    # Switch the model to evaluation mode
    classifier.eval()

    # Iterate over the test_loader
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            features = googlenet(images).squeeze()  # Extract features using GoogLeNet
            outputs = classifier(features)  # Get predictions from the classifier
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            predicted_labels.extend(predicted.cpu().numpy())  # Append predicted labels to the list
            true_labels.extend(labels.cpu().numpy())  # Append true labels to the list

    # Convert lists to NumPy arrays for easier analysis
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels) * 100
    print(f'Testing Accuracy: {accuracy:.2f}%')


# ## Question 3
# 
# - Fine Tuning the Feature Extracted Model

def custom_read_images(path, batch_size):
    """
    Read images from a specified path using PyTorch's DataLoader and apply transformations.

    Args:
    - path (str): The path to the directory containing the images.
    - batch_size (int): The batch size for DataLoader.

    Returns:
    - X (numpy.ndarray): Array of images.
    - y (numpy.ndarray): Array of corresponding labels.
    """
    data_transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])
    dataset = ImageFolder(path, transform=data_transform)
        
    data = DataLoader(dataset, batch_size=batch_size) 
    
    X = [] 
    y = []
    
    for image, label in tqdm(data):
        X.append(image) 
        y.append(label) 
        
    # Concatenate the lists of arrays along the batch dimension (axis=0)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
        
    return X, y


def augment_data():
    """
    Augment data in a DataLoader using various transformations and return an augmented DataLoader.

    Args:
    - train_loader (DataLoader): DataLoader containing the original training data.

    Returns:
    - aug_loader (DataLoader): Augmented DataLoader with transformed data for training.
    """
    
    # Create a copy of the original train_loader
    train_loader_copy = copy.deepcopy(train_loader)
    
    # Define data augmentation transformations
    augmented_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    
    # Apply the transformations to the images in train_loader
    train_loader_copy.dataset.transform = augmented_transform

    augmented_dataset = ConcatDataset([train_loader.dataset, train_loader_copy.dataset])
    aug_loader = DataLoader(augmented_dataset, batch_size=train_loader.batch_size, shuffle=True)
    return aug_loader


# Define a smaller feedforward neural network for classification
class SimpleClassifier(nn.Module):
    """
    A simple feedforward neural network for classification tasks.

    Args:
    - input_size (int): The size of the input features.
    - activation_func (str): The activation function to use. Options: "ReLU", "SiLU", "GELU", "Mish".
    - apply_dropout (str): Whether to apply dropout. Options: "Yes", "No".
    - prob (float): Dropout probability.
    - hidden_size (int): The size of the hidden layer.
    - num_classes (int): The number of output classes.

    Returns:
    - None
    """
        
    def __init__(self, input_size, activation_func, apply_dropout, prob, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # trying different activation func
        if activation_func == "ReLU": self.activation = nn.ReLU()
        elif activation_func == "SiLU": self.activation = nn.SiLU()
        elif activation_func == "GELU": self.activation = nn.GELU()
        elif activation_func == "Mish": self.activation = nn.Mish()
               
        self.apply_drop = apply_dropout
        # Adding Dropout
        self.dropout = nn.Dropout(p=prob)
        
        
        # Output Layer
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        
        if self.apply_drop == "Yes":
            x = self.dropout(x)
            
        x = self.fc2(x)
        return x

# Set device (GPU if available, otherwise CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "mps" if torch.backends.mps.is_available() else "cpu"


def extract_features():
    """
    Extract features using a pre-trained GoogLeNet model without the final classification layer.

    Returns:
    - googlenet (torch.nn.Module): Pre-trained GoogLeNet model without the final layer.
    - features (torch.Tensor): Extracted features from the images.
    - labels (torch.Tensor): Corresponding labels for the extracted features.
    """
    # Load pre-trained GoogLeNet without the final classification layer
    googlenet = models.googlenet(pretrained=True).to(device)
    googlenet = nn.Sequential(*list(googlenet.children())[:-1])  # Remove the final layer

    # Extract features using GoogLeNet
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            features = googlenet(images).squeeze()  # Remove the batch dimension
            features_list.append(features)
            labels_list.append(labels)

    # Concatenate features and labels
    features = torch.cat(features_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)
    
    return googlenet, features, labels


def evaluate_model(googlenet, classifier, test_loader):
    """
    Evaluate a classifier model using features extracted by a pre-trained GoogLeNet model.

    Args:
    - googlenet (torch.nn.Module): Pre-trained GoogLeNet model without the final layer.
    - classifier (torch.nn.Module): Classifier model to evaluate.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
    - float: Accuracy of the classifier model on the test data.
    """
    # Initialize lists to store predicted labels and ground truth labels
    predicted_labels = []
    true_labels = []

    # Switch the model to evaluation mode
    classifier.eval()

    # Iterate over the test_loader
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            features = googlenet(images).squeeze()  # Extract features using GoogLeNet
            outputs = classifier(features)  # Get predictions from the classifier
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            predicted_labels.extend(predicted.cpu().numpy())  # Append predicted labels to the list
            true_labels.extend(labels.cpu().numpy())  # Append true labels to the list

    # Convert lists to NumPy arrays for easier analysis
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels) * 100
    # print(f'Testing Accuracy: {accuracy:.2f}%')
    return accuracy



def train_model(args):
    if args.data_augment == "Yes":
        data_loader = augment_data()
        train_loader = data_loader
    
    googlenet, features, labels = extract_features()

    # Define the input size for the classifier based on the extracted features
    input_size = features.size(1)
    
    features, labels = features.to(device), labels.to(device)

    # Initialize the simple classifier and move it to the device
    classifier = SimpleClassifier(input_size, args.activation_func, args.dropout, args.prob, args.hidden_units, num_classes=10).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Trying Different Optimizers 
    if args.optimizer == "SGD": optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001) 
    elif args.optimizer == "Adam": optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001) 
    elif args.optimizer == "NAdam": optimizer = torch.optim.NAdam(classifier.parameters(), lr=0.001) 
    elif args.optimizer == "RMSprop": optimizer = torch.optim.RMSprop(classifier.parameters(), lr=0.001) 
        
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    # optimizer = torch.optim.NAdam(classifier.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    # Best Optimizer working is Adam for this problem so trying to change parameters values
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0005)
    
    run_name = f"epoch_{args.epoch}_opt_{args.optimizer}_act_{args.activation_func}_augment_{args.data_augment}_dropout_{args.dropout}_prob_{args.prob}_hu_{args.hidden_units}"


    # Train the classifier on the extracted features
    num_epochs = args.epoch # for 100 epoch this gives accuracy trian_accuracy of 89.52 %
    for epoch in range(num_epochs):
        classifier.train()  # Set the model to training mode
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        train_accuracy = correct / labels.size(0) * 100
        test_accuracy = evaluate_model(googlenet, classifier, test_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy}')
        wandb.log({"train_accuracy":train_accuracy, 'train_loss':loss.item(), 'test_accuracy':test_accuracy})
        
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()

parser = argparse.ArgumentParser(description='Parameters')

parser.add_argument('-wp', '--wandb_project', type=str, default='DL_Assignment_2',help='Project name used to track experiments in Weights & Biases dashboard.')
parser.add_argument('-we', '--wandb_entity', type=str, default='space_monkeys',help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-e', '--epoch', type=int, default=10, choices=[10, 20, 30],help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=32, choices=[16, 32, 64],help='Batch Size.')
parser.add_argument('-a', '--activation_func', type=str, default='ReLU', choices=['ReLU', 'GELU', 'SiLU', 'Mish'], help='Choices: ["ReLU", "GELU", "SiLU", "Mish"]')
parser.add_argument('-da', '--data_augment', type=str, default='No', choices=["Yes", "No"],help='Whether to apply data augmentation or not.')
parser.add_argument('-d', '--dropout', type=str, default='No', choices=['Yes', 'No'], help='Whether to apply dropout or not.')
parser.add_argument('-p', '--prob', type=float, default=0.2, choices=[0.2, 0.3], help='Probability value for dropout.')
parser.add_argument('-hu', '--hidden_units', type=int, default=256, choices=[256, 512, 1024], help='Number of hidden units.')
parser.add_argument('-o', '--optimizer', type=str, default='Adam', choices=["SGD", "Adam", "NAdam", "RMSprop"],help='Optimizer choice.')
parser.add_argument('-tdp', '--train_dataset_path', type=str, help='Path to the train dataset.')
parser.add_argument('-tep', '--test_dataset_path', type=str, help='Path to the test dataset.')

args = parser.parse_args()

train_path = args.train_dataset_path
test_path = args.test_dataset_path

X_train, y_train = custom_read_images(train_path, args.batch_size)
X_test, y_test = custom_read_images(test_path, args.batch_size)

X_train, y_train = shuffle_data(X_train, y_train)

train_loader = create_dataloader(X_train, y_train, args.batch_size)
test_loader = create_dataloader(X_test, y_test, args.batch_size)


train_model(args)