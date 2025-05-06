# Training_code_for_SegAnyNeuron

## Prerequisites

1. Install required libraries

   ```
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   
   pip install -r requirements.txt
   ```

   

2. Install pytorch-3dunet from pytorch-3dunet-master.zip

   

## Train

Run following script in terminal

```
python train.py
```

Set training dataset directory in variable **rootpath**. Set **feature_number** 0 for without feature-maps, and 7 for with feature-maps.

## Test

Run following script in terminal

```
python predict.py
```

Set test dataset directory in variable **test_path**. Set **feature_levels** 0 for without feature-maps, and 7 for with feature-maps. Set the model path in variable **model_path**. The predicted results will be saved in **test_path**/with_features and **test_path**/without_feature. 



## Pretrained Models

The pretrained models can be found in [here](https://github.com/FateUBW0227/Training_code_for_SegAnyNeuron/tree/main/pretrained_models).

## Datasets

The training datasets and test datasets can be found [here](https://zenodo.org/uploads/15293756). 

