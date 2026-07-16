import modelling.utils as utils
import modelling.data_pipeline as data_pipeline
import argparse
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.regression import CriticalSuccessIndex, MeanAbsoluteError, NormalizedRootMeanSquaredError, MeanSquaredError, R2Score
import torch
import pandas as pd
import os
from collections import defaultdict
from modelling.utils import convert_to_classification

def calculate_metrics(config, config_name, model, loader, modelling_folder, epoch, device, subset=None, subevent=None, event=None, patch=None,
                      classification=False, sensitivity=None, resolution=None):

    model.eval()
    with torch.no_grad():

        binary_prediction = False
        loss_function = "cross entropy"
        if config.get("loss_function", "cross entropy").lower()=="dice":
            binary_prediction = True
            loss_function = "dice"
        if classification:
            binary_prediction = True
        config["sensitivity"] = sensitivity
        config["resolution"] = resolution
        config["classification_evaluation"] = classification
        output_scales = ["local"] if config["only_pred_local"] else config["scales"]
        predict_feature = config.get("predict_feature", False)

        # define the metrics functions for both the binary and multi-class cases
        if binary_prediction:
            f1 = BinaryF1Score().to(device)
            precision = BinaryPrecision().to(device)
            recall = BinaryRecall().to(device)
            accuracy = BinaryAccuracy().to(device)
            csi = CriticalSuccessIndex(threshold=0.5).to(device)

        elif not predict_feature:
            f1 = MulticlassF1Score(num_classes=config["num_classes"], average=None).to(device)
            precision = MulticlassPrecision(num_classes=config["num_classes"], average=None).to(device)
            recall = MulticlassRecall(num_classes=config["num_classes"], average=None).to(device)
            accuracy = MulticlassAccuracy(num_classes=config["num_classes"], average=None).to(device)
            csi = CriticalSuccessIndex(threshold=0.5).to(device)

            metrics_functions = [f1, precision, recall, accuracy]
            metrics_names = ["f1", "precision", "recall", "accuracy"]

        else:
            mae = MeanAbsoluteError().to(device)
            mse = MeanSquaredError().to(device)
            nrmse = NormalizedRootMeanSquaredError().to(device)
            r2 = R2Score().to(device)
            metrics_functions = [mae, mse, nrmse, r2]
            metrics_names = ["mae", "mse", "nrmse", "r2"]

        # make model predictions on the dataset
        predictions, labels = defaultdict(list), defaultdict(list)
        for data in loader:
            for item in data.keys():
                if "metadata" not in item:
                    data[item] = data[item].to(device)
            model_output = model(data)
            for scale in output_scales:
                if loss_function == "dice":
                    model_output[f"{scale}_pred"] = (torch.sigmoid(model_output[f"{scale}_pred"].squeeze()) > 0.5)*1
                elif predict_feature:
                    model_output[f"{scale}_pred"] = model_output[f"{scale}_pred"].flatten()
                    data[f"{scale}_label"] = data[f"{scale}_label"].flatten()
                else:
                    model_output[f"{scale}_pred"] = torch.argmax(model_output[f"{scale}_pred"], dim=1)
                if classification:
                    model_output[f"{scale}_pred"], data[f"{scale}_label"] = convert_to_classification(config=config, pred=model_output[f"{scale}_pred"], label=data[f"{scale}_label"])
                predictions[scale].append(model_output[f"{scale}_pred"])
                labels[scale].append(data[f"{scale}_label"])

        for scale in output_scales:
            predictions[scale] = torch.concat(predictions[scale], dim=0)
            labels[scale] = torch.concat(labels[scale], dim=0)

        dataset_subset = "_".join([filter_type for filter_type in [subset, subevent, event, patch] if filter_type])
        metrics = {"config_name": [config_name], "epoch": [epoch], "dataset_subset": [dataset_subset]}
        metrics.update({key: [str(value)] for key, value in config.items()})

        # calculate metrics from the model predictions
        for scale in output_scales:
            if not predict_feature:
                for class_name, class_index in zip(["flood", "no_flood"], [2, 1]):
                    for metrics_name, metrics_function in zip(metrics_names, metrics_functions):
                        if binary_prediction and class_name=="no_flood":
                            metric = metrics_function(1-predictions[scale], 1-labels[scale])
                        else:
                            metric = metrics_function(predictions[scale], labels[scale])
                        metric = metric[class_index] if not binary_prediction else metric
                        metrics[f"{metrics_name}_{scale}_{class_name}"] = [f"{metric.item():.3f}"]
                    metrics[f"csi_{scale}_flood"] = [f"{csi((predictions[scale] == 2).int().float(), (labels[scale] == 2).int().float()).item():.3f}"]
            else:
                for metrics_name, metrics_function in zip(metrics_names, metrics_functions):
                    metric = metrics_function(predictions[scale], labels[scale])
                    metrics[f"{metrics_name}_{scale}"] = [f"{metric.item():.3f}"]

        # save the metrics results
        metrics = pd.DataFrame(metrics)
        metrics_path = f"{modelling_folder}/metrics/{config_name}.csv"
        print(f"Saving {subset} evaluation metrics to {metrics_path}")
        metrics.to_csv(metrics_path, mode="a", header=not os.path.exists(metrics_path), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('-c', '--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('-e', '--epochs', default=None, help="Load the model trained to the specified number of epochs.")
    parser.add_argument('-d', '--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('-m', '--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('-g', '--gpu', default="0", help="Specify which gpu to use. '0', '1', etc.")

    parser.add_argument('--classification', action="store_true", default=False, help="Evaluate using a classification approach.")
    parser.add_argument('--sensitivity', type=str, default="0.05", help="Specify a sensitivity (threshold) for the flood proportion.")
    parser.add_argument('--resolution', type=str, default="1", help="Specify the resolution for the classification evaluation.")

    parser.add_argument('-s', '--subset', default=None, help="Specify a data subset to calculate metrics on.")
    parser.add_argument('-b', '--subevent', default=None, help="Specify a subevent to calculate metrics on.")
    parser.add_argument('-v', '--event', default=None, help="Specify an event to calculate metrics on.")
    parser.add_argument('-p', '--patch', default=None, help="Specify a patch to calculate metrics on.")
    
    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()

    config, config_name = utils.load_config(args.config_path)
    config["batch_size"] = 4
    num_epochs = args.epochs if args.epochs else config['number_epochs']
    args.gpu = int(args.gpu) if args.gpu != "cpu" and args.gpu != "ddp" else args.gpu
    model = utils.load_model(config, rank=args.gpu, ddp=False, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{num_epochs}.pth")
    loader = data_pipeline.create_data_loader(config=config, data_folder=args.data_folder, ddp=False, subset=args.subset, subevent=args.subevent, event=args.event, patch=args.patch)
    
    args.sensitivity = [float(value) for value in args.sensitivity.replace(" ", "").split(",")]
    args.resolution = [int(value) for value in args.resolution.replace(" ", "").split(",")]

    for sensitivity in args.sensitivity:
        for resolution in args.resolution:
            calculate_metrics(config, config_name, model, loader, args.modelling_folder, epoch=num_epochs, 
                            device=args.gpu, subset=args.subset, subevent=args.subevent, event=args.event, patch=args.patch,
                            classification=args.classification, sensitivity=sensitivity, resolution=resolution)