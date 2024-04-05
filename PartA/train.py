import wandb
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms


def read_images(path):
    data_transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor()])
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


def plot_image(image_matrix):
    """
    Display an image using Matplotlib.

    Parameters:
    - image_matrix (numpy.ndarray): NumPy array containing the image data. 
                                    Should be in the format (channels, height, width).

    Returns:
    None
    """
    image = image_matrix
    # Transpose the image array from (channels, height, width) to (height, width, channels) for Matplotlib
    image = np.transpose(image, (1, 2, 0))

    # Display the image using Matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off axis labels
    plt.show()


def train_val_split(train_size:float = 0.2):
    """
    Split the training data into training and validation sets.

    Parameters:
    - train_size (float, optional): The proportion of the dataset to include in the validation set (default=0.2).

    Returns:
    - X_train (numpy.ndarray): NumPy array containing training images.
    - X_val (numpy.ndarray): NumPy array containing validation images.
    - y_train (numpy.ndarray): NumPy array containing training labels.
    - y_val (numpy.ndarray): NumPy array containing validation labels.
    """
    
    # Initialize lists to store validation and training data
    X_val = []
    y_val = []
    X = []
    y = []
    
    # 1000 because we have 1000 images of each class
    samples_per_class_val = (int) (1000 * train_size)
    
    for class_label in range(10):
        # extract indices corresponding to the current class
        class_indices = np.where(y_train==class_label)[0]
        
        # randomly select sample_per_class_val indices for validation
        val_indices = np.random.choice(class_indices, samples_per_class_val, replace=False)
        
        # append the selected val data to X_val and y_val
        X_val.extend(X_train[val_indices])
        y_val.extend(y_train[val_indices])
        
        # append the remaining data to X_train and y_train
        train_indices = np.setdiff1d(class_indices, val_indices)
        X.extend(X_train[train_indices])
        y.extend(y_train[train_indices])

    # convert python lists to np array
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X = np.array(X)
    y = np.array(y)
    
    return X, X_val, y, y_val


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


def augment_data():
    # Define data augmentation transformations
    augmented_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Apply data augmentation to the original dataset
    augmented_dataset = ConcatDataset([train_loader.dataset, train_loader.dataset])

    # Create a DataLoader for the combined dataset
    combined_loader = DataLoader(augmented_dataset, batch_size=train_loader.batch_size, shuffle=True)

    return combined_loader


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


device = "mps" if torch.backends.mps.is_available() else "cpu"

def train_step(model:torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0,0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # put data on target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calc loss (per batch) and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        # back pass
        loss.backward()

        # updating the parameters
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in data_loader:
            # send data to the target device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward pass
            test_pred = model(X_test)

            # calc loss
            test_loss += loss_fn(test_pred, y_test)

            # calc add
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # clac the test loss avg per batch
        test_loss /= len(data_loader)

        # calc the test acc avg per batch
        test_acc /= len(data_loader)
        
        return test_loss, test_acc
    

class CNN(nn.Module):
    fc_input = 1
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, activation_fn, apply_batchnorm, apply_dropout, prob, hidden_units):
        super(CNN, self).__init__()
        # Define the convolution layers
        self.conv1 = nn.Conv2d(
                               in_channels=in_channels, 
                               out_channels=num_filters[0], 
                               kernel_size=kernel_size[0], 
                               stride=1, 
                               padding=0
                              )
        self.batchnorm1 = nn.BatchNorm2d(num_filters[0])
        self.conv2 = nn.Conv2d(
                               in_channels=num_filters[0],
                               out_channels=num_filters[1], 
                               kernel_size=kernel_size[1], 
                               stride=1, 
                               padding=0
                              )
        self.batchnorm2 = nn.BatchNorm2d(num_filters[1])
        self.conv3 = nn.Conv2d(
                               in_channels=num_filters[1], 
                               out_channels=num_filters[2], 
                               kernel_size=kernel_size[2], 
                               stride=1, 
                               padding=0
                              )
        self.batchnorm3 = nn.BatchNorm2d(num_filters[2])
        self.conv4 = nn.Conv2d(
                               in_channels=num_filters[2], 
                               out_channels=num_filters[3], 
                               kernel_size=kernel_size[3], 
                               stride=1, 
                               padding=0
                              )
        self.batchnorm4 = nn.BatchNorm2d(num_filters[3])
        self.conv5 = nn.Conv2d(
                               in_channels=num_filters[3],
                               out_channels=num_filters[4], 
                               kernel_size=kernel_size[4], 
                               stride=1, 
                               padding=0
                              )
        self.batchnorm5 = nn.BatchNorm2d(num_filters[4])

        # Define activation function
        if activation_fn == "ReLU": self.activation = nn.ReLU()
        elif activation_fn == "GELU": self.activation = nn.GELU()
        elif activation_fn == "SiLU": self.activation = nn.SiLU()
        elif activation_fn == "Mish": self.activation = nn.Mish()
        
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        input_size = 227
        for i in range(5):
            input_size = input_size - kernel_size[i] + 1
            input_size = input_size // 2
        
        # Define dense layer
        self.fc1 = nn.Linear(input_size*input_size*num_filters[4], hidden_units)
        
        # Adding Dropout
        self.dropout = nn.Dropout(p=prob)
        
        # Define output layer
        self.fc2 = nn.Linear(hidden_units, 10)  # 10 output neurons for 10 classes
        
        self.apply_batchnorm = apply_batchnorm
        self.apply_dropout = apply_dropout
        

    def forward(self, x):
        # Apply convolution, activation, and max pooling layers
        x = self.pool(self.activation(self.batchnorm1(self.conv1(x)) if self.apply_batchnorm =="Yes" else self.conv1(x)))
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x)) if self.apply_batchnorm =="Yes" else self.conv2(x)))
        x = self.pool(self.activation(self.batchnorm3(self.conv3(x)) if self.apply_batchnorm =="Yes" else self.conv3(x)))
        x = self.pool(self.activation(self.batchnorm4(self.conv4(x)) if self.apply_batchnorm =="Yes" else self.conv4(x)))
        x = self.pool(self.activation(self.batchnorm5(self.conv5(x)) if self.apply_batchnorm =="Yes" else self.conv5(x)))
        
        # Flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        # Apply dense layer and output layer
        x = self.activation(self.fc1(x))
        if self.apply_dropout=="Yes": x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Apply softmax activation to the output
        return x
    

train_path = "./inaturalist_12K/train"
test_path = "./inaturalist_12K/val"

X_train, y_train = read_images(train_path)
X_test, y_test = read_images(test_path)


X_train, X_val, y_train, y_val = train_val_split(0.2) 

X_train, y_train = shuffle_data(X_train, y_train)
X_val, y_val = shuffle_data(X_val, y_val)

train_loader = create_dataloader(X_train, y_train, 32)
val_loader = create_dataloader(X_val, y_val, 32)
test_loader = create_dataloader(X_test, y_test, 32)


sweep_config = {
"name": "CNN",
"metric": {
    "name":"val_accuracy",
    "goal": "maximize"
},
"method": "bayes",
"parameters": {
        "epoch": {
            "values": [5,10]
        },
        "num_filters": {
            "values": [32, 64]
        },
        "activation_func": {
            "values": ["ReLU", "GELU", "SiLU", "Mish"]
        },
        "filter_org": {
            "values": ["same", "half", "double"]
        },
        "data_augment": {
            "values": ["Yes", "No"]
        },
        "batch_normalization": {
            "values": ["Yes", "No"]
        },
        "dropout": {
            "values": ["Yes", "No"]
        },
        "prob": {
            "values": [0.2, 0.3]
        },
        "filter_size": {
            "values": [[3,3,3,3,3], [4,4,4,4,4], [5,5,5,5,5]]
        },
        "hidden_units": {
            "values": [128, 256]
        },
    }
}


def train_cnn(train_loader, val_loader, config):
    epochs = config.epoch
    in_channels = 3
    out_channels = 10
    num_filters = [config.num_filters]
    kernel_size = config.filter_size
    activation_fn = config.activation_func
    augment = config.data_augment
    filter_org = config.filter_org
    batch_norm = config.batch_normalization
    dropout = config.dropout
    prob = config.prob
    hidden_units = config.hidden_units

    for i in range(4):
        last_value = num_filters[-1]
        if(filter_org == "same"): num_filters.append(last_value)
        elif(filter_org == "half"): num_filters.append((int)(last_value * 0.5))
        else: num_filters.append(last_value * 2)  

    if augment == "Yes":
        train_loader = augment_data()
        
    run_name = f"epoch_{epochs}_num_filters_{num_filters[0]}_act_{activation_fn}_filt_org_{filter_org}_augment_{augment}_batchnorm_{batch_norm}_dropout_{dropout}_prob_{prob}_hu_{hidden_units}"
    
    model = CNN(
                in_channels=in_channels, 
                out_channels=out_channels, 
                num_filters=num_filters, 
                kernel_size=kernel_size, 
                activation_fn=activation_fn,
                apply_batchnorm=batch_norm,
                apply_dropout=dropout,
                prob=prob,
                hidden_units=hidden_units
            )

    model.to(device)
    
    # setting up Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n------")
        train_loss, train_accuracy = train_step(
                                         model=model,
                                         data_loader=train_loader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         accuracy_fn=accuracy_fn,
                                         device=device
                                     )
        
        print(f"Train Loss: {train_loss: .5f} | Train Acc: {train_accuracy: .2f}%")

        val_loss, val_accuracy = test_step(
                                    model=model,
                                    data_loader=val_loader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    device=device
                                )

        
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        wandb.log({"val_accuracy":val_accuracy, 'val_loss':val_loss, 'train_accuracy':train_accuracy, 'train_loss':train_loss})
        
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()


def train():
    with wandb.init(project="DL_Assignment_2") as run:
        config = wandb.config
        train_cnn(train_loader, val_loader, config)

sweep_id = wandb.sweep(sweep_config, project = "DL_Assignment_2")
wandb.agent(sweep_id, train, count = 20)
wandb.finish()

