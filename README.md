# Deep Learning Models for Flood Prediction

This repository contains the code for **downloading data, training, and evaluating deep learning flood prediction models on observational data of real-world flood events, as well as analysing model behaviour** using explainable AI methodologies.

The model is input with features that describe a particular area (e.g. Sentinel 2 imagery, DEM) and the antecedent conditions (e.g. precipitation, soil moisture), represented at three different resolution scales. The model then outputs an inundation map, predicting which areas will be flooded.

# Data Creation

The data creation scripts automatically **download and process the required data**, based on the locations and times corresponding to the ground truth flooding labels. To create the dataset, firstly create an environment using *environment_data.yaml* to install the data related packages.

```conda env create --name flood-prediction-data --file environment_data.yaml```

Next follow the "preliminary set-up" instructions in *create_data.py* to pre-download the global datasets, and place the desired ground truth flooding labels in the specified folder. **To create the dataset, run:** 

``python create_data.py``

It is recommended to set the *$DATA_FOLDER* (containing the full dataset) and *$MODELLING_FOLDER* (containing the saved models and results) environmental variables, so that they do not need to be specified each time when running the scripts.

An analysis and exploration of the contents of the full dataset is available at [analysis/explore_dataset.ipynb](analysis/explore_dataset.ipynb).

# Modelling

To train and evaluate the model, firstly use the *environment_modelling.yaml* file to install the modelling related packages.

```conda env create --name flood-prediction-modelling --file environment_modelling.yaml```

Then create a *.yml* config file to define the hyperparameter values. An sample config demonstrating the available options is provided at [configs/example_config.yml](configs/example_config.yml). Next, create the data subset file and place in ``data_folder/subsets'', which associates each patch file to the train, validation or test split. An example subset csv is also provided at [configs/example_data_subset.csv](configs/example_data_subset.csv).

## Model Training

The two most important arguments to provide to the **training script** include the path to the config file and the GPU to train on; "ddp" indicates training on multiple GPUs with Distributed Data Parallel. If the environment variables pointing to the data folder and modelling folder have already been set, then they do not need to always be specified as arguments.

```python train.py --config_path=configs/example_config.yml --gpu=ddp```

The *--pretrained* argument can be pointed to the location of pre-trained model weights (as a .pth file) for fine-tuning. The *--continue_training* option can be activated to continue training a partially trained model, if it was interrupted.

## Model Evaluation

A trained model can be evaluated by either calculating **metrics** or creating **visualisations** of the predictions.

### Metrics

The dataset **subset**, **subevent**, **event** and/or individual **patch** on which to calculate metrics can be specified. As the model may have been trained (and saved) at varying number of epochs, the **epochs** argument allows you to specify which stage of the model should be used.

```python metrics.py --config_path=configs/example_config.yml --epochs=30 --event=EMSR416 --subset=train --gpu=0```

### Visualisation

**Model predictions can be visualised across an entire subevent** or for just a **single patch**. An image of the model prediction, the original label, and a comparison between prediction and label, can be saved.

```python visualise.py --config_path=configs/example_config.yml --epochs=30 --subevent=EMSR416_2019-12-15 --file_type=geotiff --gpu=0```

### ARLA

The model predictions can also be evaluated using ARLA, described further here: https://github.com/Natasha-R/ARLA-Evaluating-on-Unreliable-Labels.

```python metrics.py -config_path=configs/example_config.yml --epochs=30 --event=EMSR416 --subset=train --gpu=0 --sensitivity=1 --subpatches=4```

```python visualise.py --config_path=configs/example_config.yml --epochs=30 --subevent=EMSR416_2019-12-15 --file_type=geotiff --gpu=0 --sensitivity=1 --subpatches=4```

# Explainable AI

The model behaviour can be analysed by applying masked feature importance, replacing features with a specified value to perform rudimentary scenario modelling, or generating saliency maps from feature map gradients to explain the prediction of a particular patch.

### Feature Masking

The relevant arguments to perform feature masking include a **list of features to mask**, the **masking value**, and whether **metrics** or a **visualisation** should be then produced. The same arguments as from the *metrics.py* and *visualisation.py* scripts can also be appended as necessary. The *mask_all_features_each_scale* and *mask_all_features_all_scales* arguments sequentially masks all of the features, (and *--mask_feature_bands* for each feature band individually), to perform a full feature importance analysis.

```python xai.py --config_path=configs/example_config.yml --epochs=30 --event=EMSR416 --subset=val --mask_all_features_each_scale --mask_value=0 --metrics```

```python xai.py --config_path=configs/example_config.yml --epochs=30 --gpu=0 --subevent=EMSR416_2019-12-15 --mask_value=50 --mask_before_normalization --mask_local_features="dem" --mask_nearby_features="dem" --mask_context_features="dem" --visualise```

### Saliency Maps

To generate the saliency maps, define the **grad cam method** as *seg_grad_cam*, and provide the **patch** of interest and the **class** of interest. The **feature map** can be one of: context_down, nearby_down, local_down, local_down_first, or local_up.

```python xai.py --config_path=configs/example_config.yml --epochs=30 --grad_cam_method=seg_grad_cam --prediction_scale=local --feature_map_scale=context_down --class_of_interest=2 --patch=EMSR692_2023-09-10_000794.tif```