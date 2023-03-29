# CNN Model Optimization

Results for trained model metric on different parameters (batch_size, second last layer dense neurons):

- Result for batch size 32 and 64 with 128 neurons in second last dense layer:

<p align="center">
  <img src="Results/metric_128_32.png" alt="alt text" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="Results/metric_128_64.png" alt="alt text" width="45%"> 
</p>

- Result for batch size 32 and 64 with 256 neurons in second last dense layer:

<p align="center">
  <img src="Results/metric_256_32.png" alt="alt text" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img src="Results/metric_256_64.png" alt="alt text" width="45%"> 
</p>

## Inference

From the above four metric, the optimal parameters for training the model should be:

1. Batch size - 64
2. Epochs - 30
3. Dense layer neurons - 256
