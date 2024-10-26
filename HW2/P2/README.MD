## Running the Notebook

- Clone or download the notebook and place it in your desired directory.
- Navigate to the notebook’s directory.
- Uncomment and run the cell labeled Kaggle if you're running from platforms other than Kaggle
- Extract the dataset and ensure it’s in the correct folder structure.
- Open the notebook in Jupyter or Colab, and run the cells sequentially.

## Data Loading Scheme

- Batch size     :  256
- Context        :  80
- Input Classes     :  8631

## Architecture and Hyperparameters

During development, multiple model architectures were tested, with several hyperparameter variations for each to 
explore different configurations and determine the best-performing approach. Some of the initial models are included 
in the code as commented-out sections, providing insight into alternative architectures and configurations attempted 
during experimentation.

Previous Archtecture Explored
- a custom 5 layer CNN for feature extraction for early submission
- the number of channels for each layer were [64,128,256,512,1024]
- Each layer Accompanied by a BatchNorm and ReLu activation
- Another architecture explored was the ResNeXt.
- ResNeXt is based on the concept of residual connections allowing network to learn identity mapping
- ResNext introduced cardinality -- a split transform merge strategy to enhance feature representation

for the complete ablations, kindly check [this link to wandb public link](https://wandb.ai/DL_Busters/hw2p2-ablations/table?nw=nwusermabdulba)

### Best performing architecture & model

- I used the Sqeeze-and-Excitation ResNeXt (SEResNeXt)
- This archtecture adopts a Squeeze Excitation block for adaptive feature recalibration
- Global Average Pooling was used for summarizing global information
- The Specific archtecture used a 4 stage layer with each layer containing varying number of bottlenecks
- Similarity for verification
- Optimizer: Adam
- Metrics: Retrieval Accuracy
- Initial Learning Rate: 0.1
- Scheduler: ReduceLROnPlateau
- Batch Size: 256
- Epochs: 60

## Submitted by: mabdulba
