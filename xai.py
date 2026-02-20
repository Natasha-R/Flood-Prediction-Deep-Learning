import argparse
import os

import modelling.utils as utils
import modelling.data_pipeline as data_pipeline
from metrics import calculate_metrics
from visualise import visualise_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('-c', '--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('-e', '--epochs', default=None, help="Load the model trained to the specified number of epochs.")
    parser.add_argument('-d', '--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('-m', '--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('-g', '--gpu', default="0", help="Specify which gpu to use. '0', '1', etc.")

    parser.add_argument('--permute_local_features', type=str, default=None, help="Permute these local scale features.")
    parser.add_argument('--permute_context_features', type=str, default=None, help="Permute these context scale features.")
    parser.add_argument('--permute_basin_features', type=str, default=None, help="Permute these basin scale features.")

    parser.add_argument('-z', '--zero', action="store_true", default=False, help="Use zeroes instead of shuffling to permute the feature/patch.")

    parser.add_argument('--metrics', action="store_true", default=False, help="Calculate metrics!")
    parser.add_argument('-s', '--subset', default=None, help="Specify a data subset to calculate metrics on.")
    parser.add_argument('-b', '--subevent', default=None, help="Specify a subevent to calculate metrics on.")
    parser.add_argument('-v', '--event', default=None, help="Specify an event to calculate metrics on.")

    parser.add_argument('--visualise', action="store_true", default=False, help="Create a visualisation.")
    parser.add_argument('-p', '--patch', default=None, help="Visualise model predictions of the specified patch.")
    parser.add_argument('-a', '--scale', default="local", help="Define the scale for the visualisation.")
    parser.add_argument('-f', '--file_type', default="geotiff", help="Save the image file as either a 'geotiff' or 'png'.")
    parser.add_argument('-o', '--pred_only', action="store_true", default=False, help="Save only the prediction, and not label & comparison")
    parser.add_argument('--test_border', default=0, help="Print a border around the test set images, with the given pixel size")
    
    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    logger = utils.get_logger()

    permute_features = {}
    for scale, scale_permute_features in zip(["local", "context", "basin"], [args.permute_local_features, args.permute_context_features, args.permute_basin_features]):
         if scale_permute_features: permute_features[scale] = scale_permute_features.replace(" ", "").split(",")

    config, config_name = utils.load_config(args.config_path, logger)
    config["batch_size"] = 4
    config["zero_out"] = args.zero
    config["permute_features"] = permute_features
    num_epochs = args.epochs if args.epochs else config['number_epochs']
    args.gpu = int(args.gpu)
    permute_features_str = str(permute_features)
    for punctuation in ["'", "{", "}", " ", "_", ":"]:
        permute_features_str = permute_features_str.replace(punctuation, "")
    permute_features_str = permute_features_str.replace(",", "_")
            
    model = utils.load_model(config, rank=args.gpu, logger=logger, ddp=False, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{num_epochs}.pth")
    loader = data_pipeline.create_data_loader(config=config, data_folder=args.data_folder, ddp=False, 
                                              subset=args.subset, subevent=args.subevent, event=args.event,
                                              permute_features=permute_features)

    if args.metrics:
        calculate_metrics(config, config_name, model, loader, args.modelling_folder, epoch=num_epochs, 
                          logger=logger, device=args.gpu, subset=args.subset, subevent=args.subevent, event=args.event)
        
    if args.visualise:
        if permute_features:
            config_name += f"_permute{permute_features_str}_zero{args.zero}"
        visualise_predictions(config=config, config_name=config_name, model=model, num_epochs=num_epochs, data_folder=args.data_folder, modelling_folder=args.modelling_folder, 
                              device=args.gpu, logger=logger, subevent=args.subevent, file_type=args.file_type, scale=args.scale, patch=args.patch, 
                              pred_only=args.pred_only, test_border=int(args.test_border), permute_features=permute_features)


