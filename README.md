# Deep Learning Models for Flood Prediction

This repository contains the code for **downloading data, training and evaluating deep learning flood prediction models, as well as analysing the model behaviour** using explainable AI methodologies.

The model is input with features that describe a particular area (e.g. Sentinel 2 imagery, DEM) and the antecedent conditions (e.g. precipitation, soil moisture), represented at three different resolution scales. The model then outputs an inundation map, predicting which areas will be flooded.

# Data Creation

The data creation scripts automatically **download and process the required data**, based on the locations and times corresponding to the ground truth flooding labels. To create the dataset, firstly create an environment using *environment_data.yaml* to install the data related packages.

```conda create --name flood-prediction-data --file environment_data.yaml```

Next follow the "preliminary set-up" instructions in *create_data.py* to pre-download the necessary global datasets, and place the desired ground truth flooding labels in the specified folder. **To create the dataset, run:** 

``python create_data.py``

It is recommended to set the *$DATA_FOLDER* (containing the full dataset) and *$MODELLING_FOLDER* (containing the saved models and results) environmental variables, so that they do not need to be specified each time when running the scripts.

An analysis and exploration of the contents of the full dataset is available at [analysis/explore_dataset.ipynb](analysis/explore_dataset.ipynb).

# Modelling

To train and evaluate the model, firstly use the *environment_modelling.yaml* file to install the modelling related packages.

```conda create --name flood-prediction-modelling --file environment_modelling.yaml```

Then create a *.yml* config file to define the model architecture, hyperparameters, and data set-up. An sample config demonstrating the available changeable options is provided at [configs/example_config.yml](configs/example_config.yml).

## Model Training

The two most important arguments to provide to the **training script** include the path to the config file and the GPU to train on - "ddp" indicates training on multiple GPUs with Distributed Data Parallel. If the environment variables pointing to the data folder and modelling folder have already been set, then they do not need to be specified as arguments.

```python train.py --config_path=configs/example_config.yml --gpu=ddp```

## Model Evaluation

The trained model can then be evaluated by either calculating **metrics** or creating **visualisations** of the predictions.

### Metrics

The dataset **subset**, **subevent**, **event** and/or individual **patch** on which to calculate metrics can be specified. As the model may have been trained (and saved) at varying number of epochs, the **epochs** argument allows you to specify which model should be used.

```python metrics.py --config_path=configs/example_config.yml --epochs=30 --subset=val```

```python metrics.py --config_path=configs/example_config.yml --epochs=30 --event=EMSN066 --subset=test```

### Visualisation

**Model predictions can be visualised across an entire subevent** (first example) or for just a **single patch** (second example), at the the provided scale (default "local"). The image can be saved as either a geotiff or png. "pred_only" creates only the model prediction, and not also the original label and comparison between prediction and label. "test_border" overlays a border of the provided pixel width around the validation/test set patches, in order to distinguish them from training patches.

```python visualise.py --config_path=configs/example_config.yml --epochs=30 --subevent=EMSN066_2020-01-11 --scale=local --file_type=png --pred_only --test_border=10```

```python visualise.py --config_path=configs/example_config.yml --epochs=30 --patch=EMSN066_2020-01-11_000038.tif --scale=basin --file_type=geotiff```

# Explainable AI

The model behaviour can be analysed by masking features, masking a portion of the image, or using saliency maps via gradients to explain the prediction of a particular region.

### Feature Masking

The relevant arguments to perform feature masking include a **list of features to mask**, and whether **metrics** or a **visualisation** should be then produced. The same arguments as from the *metrics.py* and *visualisation.py* scripts can then be appended as necessary.

```python xai.py --config_path=configs/example_config.yml --epochs=30 --mask_local_features="sentinel2, sentinel1" --mask_context_features="sentinel2, sentinel1" --mask_basin_features="sentinel2, sentinel1"--metrics --subset=val```

```python xai.py --config_path=configs/example_config.yml --epochs=30 --mask_local_features="sentinel2, sentinel1" --visualise --subevent=EMSN066_2020-01-11 --scale=local --file_type=png```

### Image Masking

To **mask a portion of the image**, provide the "mask_box" argument, the coordinates for the "box_top_left" and "box_bottom_right", and the specific **patch** to be masked, at which scale. As with the feature masking, either metrics or a visualisation can then be produced.

```python xai.py --config_path=configs/example_config.yml --epochs=30 --mask_box --box_top_left="0, 0" --box_bottom_right="100, 100" --scale=local --patch=EMSN066_2020-01-11_000038.tif --visualise --pred_only --file_type=png```

Feature masking and image masking can also be performed simultaneously, by simply providing both sets of arguments.

### Saliency Maps

Specify the "grad_cam_method" to generate the saliency map - either "seg_grad_cam" or "seg_xres_cam". Then provide the "prediction_scale" that you want to explain, and the "feature_map_scale" to generate the explanation from. The "class_of_interest" to explain can be either 1 for non-flood or 2 for flood. "box_top_left" and "box_bottom_right" defines the region to be explained from the given "patch".

```python xai.py --config_path=configs/example_config.yml --epochs=30 --grad_cam_method=seg_grad_cam --prediction_scale=local --feature_map_scale=local --box_top_left="50, 50", --box_bottom_right="100, 100" --class_of_interest=2 --patch=EMSN066_2020-01-11_000038.tif```