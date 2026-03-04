import argparse
import os
import modelling.utils as utils
import modelling.data_pipeline as data_pipeline
from metrics import calculate_metrics
from visualise import visualise_predictions
from modelling.seg_grad_cam import seg_grad_cam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise the model's predictions.")
    parser.add_argument('-c', '--config_path', required=True, help="The path to the configuration file to use.")
    parser.add_argument('-e', '--epochs', default=None, help="Load the model trained to the specified number of epochs.")
    parser.add_argument('-d', '--data_folder', default=os.environ["DATA_FOLDER"], help="The path to the dataset folder.")
    parser.add_argument('-m', '--modelling_folder', default=os.environ["MODELLING_FOLDER"], help="The path to the modelling folder.")
    parser.add_argument('-g', '--gpu', default="0", help="Specify which gpu to use. '0', '1', etc.")

    parser.add_argument('--mask_local_features', type=str, default=None, help="Mask these local scale features.")
    parser.add_argument('--mask_context_features', type=str, default=None, help="Mask these context scale features.")
    parser.add_argument('--mask_basin_features', type=str, default=None, help="Mask these basin scale features.")
    parser.add_argument('--permute', action="store_true", default=False, help="Use permute by shuffling instead of zero-out the feature/patch.")

    parser.add_argument('--metrics', action="store_true", default=False, help="Calculate metrics.")
    parser.add_argument('--visualise', action="store_true", default=False, help="Create a visualisation.")

    parser.add_argument('--mask_box', action="store_true", default=False, help="Mask the provided region when generating a prediction.")
    parser.add_argument('--box_top_left', type=str, default=None, help="Define a box on the patch image, in pixels.")
    parser.add_argument('--box_bottom_right', type=str, default=None, help="Define a box on the patch image, in pixels.")

    parser.add_argument('--grad_cam_method', default=None, help="Specify the method: 'seg_grad_cam' or 'seg_xres_cam'.")
    parser.add_argument('--prediction_scale', default="local", help="The prediction scale that we want to explain.")
    parser.add_argument('--feature_map_scale', default="local", help="The feature map scale that we are generating the explanation from.")
    parser.add_argument('--class_of_interest', default=2, type=int, help="The class that we want to generate an explanation of (0 for non-flood, 1 for flood).")

    parser.add_argument('-f', '--file_type', default="geotiff", help="Save the image file as either a 'geotiff' or 'png'.")
    parser.add_argument('-o', '--pred_only', action="store_true", default=False, help="Save only the prediction, and not label & comparison")
    parser.add_argument('--test_border', default=0, help="Print a border around the test set images, with the given pixel size")

    parser.add_argument('-s', '--subset', default=None, help="Specify a data subset.")
    parser.add_argument('-b', '--subevent', default=None, help="Specify a subevent.")
    parser.add_argument('-v', '--event', default=None, help="Specify an event.")
    parser.add_argument('-p', '--patch', default=None, help="Specify the patch.")
    parser.add_argument('-a', '--scale', default="local", help="Define the scale.")

    args = parser.parse_args()

    utils.check_paths(args)
    utils.check_cuda()
    logger = utils.get_logger()
    mask_features, mask_patch = {}, {}
    config, config_name = utils.load_config(args.config_path, logger)
    config["batch_size"] = 4
    config["permute"] = args.permute
    num_epochs = args.epochs if args.epochs else config['number_epochs']
    args.gpu = int(args.gpu)

    if args.mask_local_features or args.mask_context_features or args.mask_basin_features:
        for scale, scale_mask_features in zip(["local", "context", "basin"], [args.mask_local_features, args.mask_context_features, args.mask_basin_features]):
            if scale_mask_features: mask_features[scale] = scale_mask_features.replace(" ", "").split(",")
        new_config_name = config_name + "_" + utils.mask_to_string(mask_features)

    if args.mask_box:
        mask_patch = {"scale": args.scale, 
                      "top_left":args.box_top_left.replace(" ", "").split(","),
                      "bottom_right": args.box_bottom_right.replace(" ", "").split(",")}
        new_config_name = config_name + "_" + utils.mask_to_string(mask_patch)

    model = utils.load_model(config, rank=args.gpu, logger=logger, ddp=False, pretrained_path=f"{args.modelling_folder}/models/{config_name}_{num_epochs}.pth")

    if args.grad_cam_method:
        new_config_name = f"{config_name}_{num_epochs}epochs_{args.grad_cam_method}_PRED{args.prediction_scale}_MAP{args.feature_map_scale}_class{args.class_of_interest}_({args.box_top_left})({args.box_bottom_right})_{args.patch}"
        args.box_top_left = tuple([int(value) for value in args.box_top_left.replace(" ", "").split(",")])
        args.box_bottom_right = tuple([int(value) for value in args.box_bottom_right.replace(" ", "").split(",")])
        seg_grad_cam(config=config, config_name=new_config_name, data_folder=args.data_folder, modelling_folder=args.modelling_folder, model=model, device=args.gpu, method=args.grad_cam_method, logger=logger,
                     patch=args.patch, target_map=args.feature_map_scale, target_pred=args.prediction_scale, box_top_left=args.box_top_left, box_bottom_right=args.box_bottom_right, class_of_interest=args.class_of_interest)
    
    else:
        if args.metrics:
            loader = data_pipeline.create_data_loader(config=config, data_folder=args.data_folder, ddp=False, subset=args.subset, subevent=args.subevent, event=args.event,
                                                      mask_features=mask_features, mask_patch=mask_patch)
            calculate_metrics(config, new_config_name, model, loader, args.modelling_folder, epoch=num_epochs, 
                            logger=logger, device=args.gpu, subset=args.subset, subevent=args.subevent, event=args.event)
            
        if args.visualise:
            visualise_predictions(config=config, config_name=new_config_name, model=model, num_epochs=num_epochs, data_folder=args.data_folder, modelling_folder=args.modelling_folder, 
                                device=args.gpu, logger=logger , subevent=args.subevent, file_type=args.file_type, scale=args.scale, patch=args.patch, 
                                pred_only=args.pred_only, test_border=int(args.test_border), mask_features=mask_features, mask_patch=mask_patch)
        
