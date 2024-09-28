## Running the Notebook
 - Clone or download the notebook and place it in your desired directory.
 - Navigate to the notebook’s directory.
 - Uncomment and run the cell labeled Kaggle if you're running from platforms other than Kaggle
 - Extract the dataset and ensure it’s in the correct folder structure.
 - Open the notebook in Jupyter or Colab, and run the cells sequentially.
 
 
## Data Loading Scheme
 - Batch size     :  1024
 - Context        :  40
 - Input size     :  2268
 - Output symbols :  42
 - Train dataset samples = 36091157, batches = 35246
 - Validation dataset samples = 1928204, batches = 1884
 - Test dataset samples = 1934138, batches = 1889
 
 ## Architecture and Hyperparameters
for the complete ablations, kindly check [this link to wandb public link](https://wandb.ai/mabdulba-carnegie-mellon-university/hw1p2/table)

 ### Best performing architecture & model

   - Type: Wide and Deep MLP Network (Contrasting Width) 
   - Layers: [Input layer (128 units), Hidden layer 1 (2268 units), Hidden layer 5, Output layer(42)]
   - Activation Functions: GELU
   - Loss Function: Categorical Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy
   - Initial Learning Rate: 0.001
   - Scheduler: ReduceLROnPlateau
   - Batch Size: 1024
   - Context: 40
   - Epochs: 40
   
## Submitted by: mabdulba