import modelling.utils as utils
import modelling.data_pipeline as data_pipeline
import argparse
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics.segmentation import MeanIoU
import torch
import pandas as pd
import os

def calculate_metrics(config, config_name, model, loader, modelling_folder, epoch, logger, device, subset="val_other", subevent=None, event=None):

    # define the metrics functions
    class_f1 = MulticlassF1Score(num_classes=config["num_classes"], average=None).to(device)
    class_precision = MulticlassPrecision(num_classes=config["num_classes"], average=None).to(device)
    class_recall = MulticlassRecall(num_classes=config["num_classes"], average=None).to(device)
    class_accuracy = MulticlassAccuracy(num_classes=config["num_classes"], average=None).to(device)
    class_iou = MeanIoU(num_classes=config["num_classes"], per_class=True, input_format="index").to(device)
    metrics_functions = [class_f1, class_precision, class_recall, class_accuracy]
    metrics_names = ["f1", "precision", "recall", "accuracy"]

    # make model predictions on the dataset
    predictions, labels = [], []
    for data in loader:
        local_features = data["local_features"]
        local_label = data["local_label"].to(device) 
        model_output = model(local_features)
        predicted_class = torch.argmax(model_output, dim=1)
        predictions.append(predicted_class)
        labels.append(local_label)
    predictions = torch.concat(predictions, dim=0)
    labels = torch.concat(labels, dim=0)

    if config["separate_flood_trace_label"]:
       class_names = ["flooded_area",  "flood_trace", "no_flood"] 
       class_indices = [3, 2, 1]
    else: 
        class_names = ["flood", "no_flood"]
        class_indices = [2, 1]

    dataset_subset = "_".join([filter_type for filter_type in [subset, subevent, event] if filter_type])
    metrics = {"config_name": [config_name], "epoch": [epoch], "dataset_subset": [dataset_subset]}
    metrics.update({key: [str(value)] for key, value in config.items()})

    # calculate metrics from the model predictions
    for class_index, class_name in zip(class_indices, class_names):
        for metrics_name, metrics_function in zip(metrics_names, metrics_functions):
            metric = metrics_function(predictions, labels)[class_index]
            metrics[f"{metrics_name}_{class_name}"] = [f"{metric.item():.3f}"]
        metric = class_iou(predictions.long().unsqueeze(-1), labels.long().unsqueeze(-1))[class_index]
        metrics[f"iou_{class_name}"] = [f"{metric.item():.3f}"]

    # save the metrics results
    metrics = pd.DataFrame(metrics)
    metrics_path = f"{modelling_folder}/metrics/{config_name}.csv"
    logger.info(f"Saving evaluation metrics to {metrics_path}")
    metrics.to_csv(metrics_path, mode="a", header=not os.path.exists(metrics_path), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('--gpu', default="cuda:0", help="Specify which gpu to use. 'cuda', 'cuda:0', 'cuda:1', etc.")

    parser.add_argument('--subset', default="validation", help="Specify a data subset to calculate metrics on.")
    parser.add_argument('--subevent', default=None, help="Specify a subevent to calculate metrics on.")
    parser.add_argument('--event', default=None, help="Specify an event to calculate metrics on.")
    
    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    logger = utils.get_logger()

    config, config_name = utils.load_config(args.config_path, logger)
    model = utils.load_model(config, device=args.gpu, logger=logger, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{config['number_epochs']}.pth")
    loader = data_pipeline.create_data_loader(config=config, data_folder=args.data_folder, subset=args.subset, subevent=args.subevent, event=args.event)

    calculate_metrics(config, config_name, model, loader, args.modelling_folder, epoch=config["number_epochs"], logger=logger, device=args.gpu,
                      subset=args.subset, subevent=args.subevent, event=args.event)