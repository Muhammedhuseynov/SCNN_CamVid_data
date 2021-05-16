Model code from this repos https://github.com/harryhan618/SCNN_Pytorch
I changed some places for my own tasks.

And I used this library https://github.com/qubvel/segmentation_models.pytorch for "Loss function","IOU score" and "Custom Dataset"

--- What is new?
You can train it with 1 channel imgs and 3 channels
Ease to make custom dataset

# for example I used CamVid Dataset check this link https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

----

For Train:
Change train_process.py/ Line 76: Put your own classes and change path to dataset and Carefully read comments!
Change data_process.pt/ Line 38: Put your classes and Carefully read comments!

for Visualize data:
Change visualize_data.py/ Line 53 Put your own classes and change path to your classes! Carefully read Comments!
read comment on Line 55

for Test Model:
Change modelTest_process.py/ Line 14 Put your own classes and change path to your classes! Carefully read Comments!
And read comment on Line 19

# SCNN_withCamVid_dataset
# SCNN_with_CamVid_dataset
