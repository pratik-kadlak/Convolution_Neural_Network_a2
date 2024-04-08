# CS6910: Deep Learning Assignment 2

# Part A

## Overview
This repository contains code and resources for building a small CNN model with 5 convolution layers, each followed by an activation and max-pooling layer. The model includes a dense layer and an output layer with 10 neurons, suitable for classifying images in the iNaturalist dataset. The code is implemented in a Jupyter Notebook using PyTorch and includes functionality for hyperparameter tuning using W&B Sweeps.

## Contents
- `PartA/part-a.ipynb/`: Directory containing the Jupyter Notebook with the CNN model implementation and hyperparameter tuning.
- `train_PartA.py`: Contains Model that runs on the parameters given by the user
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

## Supported Arguments
Here's a table representation of the argparse arguments for your project:

| Argument              | Shorthand | Type     | Default         | Choices                                          | Description                                                                                                      |
|-----------------------|-----------|----------|-----------------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| wandb_project         | -wp       | str      | DL_Assignment_2 | -                                                | Project name used to track experiments in Weights & Biases dashboard.                                            |
| wandb_entity          | -we       | str      | space_monkeys   | -                                                | Wandb Entity used to track experiments in the Weights & Biases dashboard.                                         |
| epochs                | -e        | int      | 10              | -                                                | Number of epochs to train neural network.                                                                         |
| activation            | -a        | str      | ReLU            | ReLU, GELU, SiLU, Mish                           | Choices for activation function: "ReLU", "GELU", "SiLU", "Mish".                                                 |
| num_filters           | -nf       | int      | 32              | -                                                | Number of filters in starting layer.                                                                              |
| filter_org            | -fo       | str      | same            | same, half, double                               | Choices for filter organization: "same", "half", "double".                                                       |
| augment               | -da       | str      | No              | Yes, No                                          | Choices for data augmentation: "Yes", "No".                                                                       |
| batch_norm            | -bn       | str      | Yes             | Yes, No                                          | Choices for batch normalization: "Yes", "No".                                                                     |
| dropout               | -d        | str      | Yes             | Yes, No                                          | Choices for dropout: "Yes", "No".                                                                                 |
| prob                  | -p        | float    | 0.3             | 0.2, 0.3                                         | Choices for dropout probability: 0.2, 0.3.                                                                        |
| filter_size           | -fs       | str      | [3,3,3,3,3]     | [3,3,3,3,3], [4,4,4,4,4], [5,5,5,5,5]            | Choices for filter size: "[3,3,3,3,3]", "[4,4,4,4,4]", "[5,5,5,5,5]".                                           |
| hidden_units          | -hu       | int      | 128             | 128, 256                                         | Choices for hidden units: 128, 256.                                                                              |
| train_dataset_path    | -tdp      | str      | -               | -                                                | Path to the train dataset.                                                                                       |
| test_dataset_path     | -tep      | str      | -               | -                                                | Path to the test dataset.                                                                                        |
| mode                  | -m        | str      | normal          | normal, plot                                     | Choices for mode: "normal", "plot".  |

Example: `python train_PartA.py -wp DL_Assignment_2 -we space_monkeys -e 10 -a ReLU -nf 32 -fo same -da No -bn Yes -d Yes -p 0.3 -fs '[3,3,3,3,3]' -hu 128 -tdp /path/to/train -tep path/to/test -m best`

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
- `train_PartB.py`: Googlenet model run on parameters passed by command line
- `requirements.txt`: File listing the required Python libraries and versions.

### Running the Code
1. Clone the repository : https://github.com/pratik-kadlak/Convolution_Neural_Network_a2.git
2. Install the required libraries listed in `requirements.txt`.
3. Open and run the `PartB/part-b.ipynb` notebook in a Jupyter environment.

### Results and Analysis
- The notebook includes detailed results and analysis for each model variant.
- Metrics such as accuracy, loss, and model performance are evaluated.

### Supported Commands
Here's the GitHub markdown table for the argparse arguments:

| Argument              | Shorthand | Type   | Default        | Choices               | Description                                                                                               |
|-----------------------|-----------|--------|----------------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| --wandb_project       | -wp       | str    | DL_Assignment_2 |                      | Project name used to track experiments in Weights & Biases dashboard.                                      |
| --wandb_entity        | -we       | str    | space_monkeys  |                      | Wandb Entity used to track experiments in the Weights & Biases dashboard.                                  |
| --epoch               | -e        | int    | 10             | 10, 20, 30            | Number of epochs to train neural network.                                                                 |
| --batch_size          | -b        | int    | 32             | 16, 32, 64            | Batch Size.                                                                                               |
| --activation_func     | -a        | str    | ReLU           | ReLU, GELU, SiLU, Mish | Choices for activation function: "ReLU", "GELU", "SiLU", "Mish".                                          |
| --data_augment        | -da       | str    | No             | Yes, No                | Whether to apply data augmentation or not.                                                                 |
| --dropout             | -d        | str    | No             | Yes, No                | Whether to apply dropout or not.                                                                          |
| --prob                | -p        | float  | 0.2            | 0.2, 0.3               | Probability value for dropout.                                                                            |
| --hidden_units        | -hu       | int    | 256            | 256, 512, 1024        | Number of hidden units.                                                                                    |
| --optimizer           | -o        | str    | Adam           | SGD, Adam, NAdam, RMSprop | Optimizer choice: "SGD", "Adam", "NAdam", "RMSprop".                                                      |
| --train_dataset_path  | -tdp      | str    |                |                       | Path to the train dataset.                                                                                 |
| --test_dataset_path   | -tep      | str    |                |                       | Path to the test dataset.                                                                                  |

Example: `python train_PartB.py -wp My_Project -we My_Entity -e 20 -b 64 -a Mish -da Yes -d Yes -p 0.2 -hu 512 -o Adam -tdp /path/to/train_data -tep /path/to/test_data`

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
