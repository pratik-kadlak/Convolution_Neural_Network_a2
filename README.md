# CS6910: Deep Learning Assignment 2

# Part A

## Overview
This repository contains code and resources for building a small CNN model with 5 convolution layers, each followed by an activation and max-pooling layer. The model includes a dense layer and an output layer with 10 neurons, suitable for classifying images in the iNaturalist dataset. The code is implemented in a Jupyter Notebook using PyTorch and includes functionality for hyperparameter tuning using W&B Sweeps.

## Contents
- `PartA/part-a.ipynb/`: Directory containing the Jupyter Notebook with the CNN model implementation and hyperparameter tuning.
- `requirements.txt`: List of required Python packages.

## Getting Started
1. Clone the repository:

```bash
git clone https://github.com/pratik-kadlak/Convolution_Neural_Network_a2.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data:
   - Place the iNaturalist dataset or your custom dataset in the `data/` directory.
   - Here is the link for the iNaturalist dataset: https://storage.googleapis.com/wandb_datasets/nature_12K.zip

## Running the Code
1. Open and run the Jupyter Notebook `PartA/part-a.ipynb`.
2. Follow the instructions in the notebook to:
   - Run Each Cell 
   - Train the model on the dataset.
   - Use W&B Sweeps for hyperparameter tuning.

## Hyperparameter Tuning
Use W&B Sweeps for automated hyperparameter tuning. Modify `sweep_config.yaml` in the notebook to define the search space for hyperparameters.

# Part B
## GoogleNet on iNaturalist Dataset

### Overview
This project explores the usage of the GoogleNet pre-trained model on the iNaturalist dataset. The goal is to build and evaluate different models based on GoogleNet for classification tasks.

### Models Implemented
1. **Freezed Last Layer Model**:
   - The weights of all layers in GoogleNet are frozen except for the last layer.
   - The last layer's weights are updated to fit the specific classification task of the iNaturalist dataset.

2. **Partial Freezing Model**:
   - Certain layers (up to k layers) in GoogleNet are frozen while the remaining layers are trainable.
   - This allows for more flexibility in adapting GoogleNet to the dataset.

3. **Feature Extraction Model**:
   - Features are extracted from GoogleNet using a pre-trained model.
   - These features are then used in a small classifier to perform classification tasks.

### Implementation Details
- The models are implemented in a Jupyter Notebook using PyTorch.
- The code is designed to be flexible, allowing for changes in the number of filters, filter sizes, activation functions, and number of neurons in the dense layer.
- Sweep functionality is implemented to automate model runs with various configurations, making hyperparameter tuning more efficient.

### File Structure
- `part-b.ipynb.ipynb`: Jupyter Notebook containing the code for the models and experiments.
- `requirements.txt`: File listing the required Python libraries and versions.

### Running the Code
1. Clone the repository : https://github.com/pratik-kadlak/Convolution_Neural_Network_a2.git
2. Install the required libraries listed in `requirements.txt`.
3. Open and run the `PartB/part-b.ipynb` notebook in a Jupyter environment.

### Results and Analysis
- The notebook includes detailed results and analysis for each model variant.
- Metrics such as accuracy, loss, and model performance are evaluated.

### Future Work
- Explore additional model variants and architectures for better performance.
- Experiment with different optimization techniques and learning rates.
- Conduct more extensive hyperparameter tuning and experimentation.


## Here is the Report link of the experiments run by me
Report Link: https://wandb.ai/space_monkeys/DL_Assignment_2/reports/CS6910-Assignment-2--Vmlldzo3NDM3OTgw


## Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

### Contributors
- Pratik Kadlak
  
## Support
For questions or assistance, contact pratikvkadlak2001@gmail.com.


---
