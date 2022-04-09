from collections import namedtuple
from glob import glob
from os import path 
import argparse
import json
import traceback

import numpy as np
from PIL import Image
from numpy.lib.function_base import average


#########################
## Global Variables :( ##
#########################
_suppress_dimension_warning = False
_suppress_unique_values_warning = False

#################
## Definitions ##
#################

Label = namedtuple('Label', [
    'id',   # 0-based integer allowing enumeration and representing training ID
    'name', # String name of class, e.g. car
    'rgb'   # RGB value of class in image as int tuple
])

#############
## Methods ##
#############
def create_confusion_matrix(labels):
    """Create confusion matrix for labels.\n
    returns: numpy array representing confusion matrix"""
    
    # Define confusion matrix size by largest label ID
    max_id = max([label.id for label in labels])
    
    matrix_dim = max_id+1

    return np.zeros(shape=(matrix_dim, matrix_dim),dtype=np.ulonglong)

def rgb_to_label(img, label_map):
    """Converts RGB image to masked image.
    Adapted From: https://stackoverflow.com/a/62170172/15386165"""
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]]) # FIXME understand why this is done

    # Get unique "ID" encodings for each pixel in image
    img_id = img.dot(W).squeeze(-1) 

    # Get each unique encoding
    unique_values = np.unique(img_id)

    if len(unique_values) > len(label_map) * 2:
        global _suppress_unique_values_warning
        if not _suppress_unique_values_warning:
            print(f"WARNING: Large number of unique RGB values ('{len(unique_values)}') in input image. This may be a sign there is a problem with your input images, and will affect the speed of this script.")
            _suppress_unique_values_warning = True

    # Generate label mask
    mask = np.zeros(img_id.shape)

    for i, c in enumerate(unique_values):
        try:
            # Perform lookup in color map
            mask[img_id==c] = label_map[tuple(img[img_id==c][0])] 
        except:
            mask[img_id==c] = -1
    
    return mask.astype(np.int32)

def evaluate_pair(ground_truth_filename:np.ndarray, prediction_filename:np.ndarray, confusion_matrix, labels):
    """Evaluate ground truth & prediction image pair, adding results to confusion matrix"""
    ## Load images
    ground_truth_pil = Image.open(ground_truth_filename).convert('RGB')
    ground_truth = np.asarray(ground_truth_pil)

    prediction_pil = Image.open(prediction_filename).convert('RGB')
    prediction  = np.asarray(prediction_pil)

    # TODO for now we are evaluating on RGB images. The original script evaluates on grayscale ID=RGB value images, which may be much faster. This really depends on the speed of the encoding into labels done here.
    ## Sanity check image sizes
    if (prediction.shape[0] != ground_truth.shape[0]) or (prediction.shape[1] != ground_truth.shape[1]):
        # Raise warning the first time this happens
        global _suppress_dimension_warning
        if not _suppress_dimension_warning:
            print(f"WARNING: Dimensions of input ground truth '{ground_truth_filename}' and prediction '{prediction_filename}' are not equal. This warning will only be shown once.")
            _suppress_dimension_warning = True

        # Reshape the prediction to same size as ground truth
        prediction_pil = prediction_pil.resize([ground_truth_pil.width, ground_truth_pil.height], resample=Image.NEAREST)
        prediction  = np.asarray(prediction_pil)

    if len(prediction.shape) != 3:
        raise ValueError("Predicted image does not have 3 dimensions (width, height, colour).")

    ## Encode the segmentation maps
    # Generate label map (converts RGB to label class value)
    label_map = {tuple(label.rgb):label.id for label in labels}
    label_ids = [label.id for label in labels]

    # Convert images to label format
    ground_truth_label = rgb_to_label(ground_truth, label_map)
    prediction_label = rgb_to_label(prediction, label_map)

    ## Compare ground truth & prediction, add to confusion matrix
    # Get every unique combination of ground-truth class/prediction class for a pixel in the pair
    unique_encoding_seed = max(ground_truth_label.max(), prediction_label.max()).astype(np.int32) + 1
    unique_encoded = (ground_truth_label.astype(np.int32) * unique_encoding_seed) + prediction_label

    # Sum up all unique combinations and get their counts
    unique_values, unique_value_counts = np.unique(unique_encoded, return_counts=True)
    
    # Add results to confusion matrix
    for value, count in zip(unique_values, unique_value_counts):
        # Get prediction and ground truth IDs back from encoding
        pred_label = value % unique_encoding_seed
        gt_label = int((value - pred_label)/unique_encoding_seed)

        # Show warning if unknown label was predicted
        if not pred_label in label_ids:
            print(f"Unknown label predicted with id {gt_label}")

        # Skip if ground truth is unknown label (e.g. void)
        if not gt_label in label_ids:
            continue

        # Add ground truth/prediction combination to confusion matrix
        confusion_matrix[gt_label,pred_label] += count
    
def display_confusion_matrix(confusion_matrix:np.ndarray, labels):
    """Displays the confusion matrix in a formatted string."""
    confusion_matrix_transpose = confusion_matrix.copy().transpose()

    # fcm = format confusion matrix
    fcm = []
    
    fcm_labels = [label.name for label in labels]
    fcm_labels.insert(0, "")
    fcm.append(fcm_labels)

    # Add relevant parts of full confusion matrix to formatted confusion matrix
    for label_row in labels:
        fcm_row = []
        fcm_row.append(label_row.name)
        for label_col in labels:
            fcm_row.append(confusion_matrix_transpose[label_row.id, label_col.id])
    
        fcm.append(fcm_row)

    longest_item = max([len(max([str(i) for i in row], key=len)) for row in fcm])
    item_length = longest_item + 1

    ## Compile matrix into string
    fcm_string = ""
    for row in fcm:
        fcm_string += ''.join([str(item).rjust(item_length) for item in row])
        fcm_string += '\n'
    
    return fcm_string

def display_ious(confusion_matrix, labels):
    """Return per-class and mean IoU scores in a formatted string, based on input confusion matrix
    NOTE: Mean IoU score is currently not weighted"""
    
    iou_string = ""

    iou_string += f"{'Class'.rjust(10)}{'IoU'.rjust(25)}" + '\n' # FIXME adjust by longest label name
    for label in labels:
        print(f"{label.name.rjust(10)}{str(get_iou(label.id, confusion_matrix)).rjust(25)}")

    ## Calculate Mean IoU
    valid_ious = []
    for label in labels:
        iou = get_iou(label.id, confusion_matrix)
        if not np.isnan(iou):
            valid_ious.append(iou)

    mean_iou = np.mean(valid_ious) # FIXME does the mean have to be weighted???

    iou_string += f"{'Mean'.rjust(10)}{str(round(mean_iou, 4)).rjust(25)}" + '\n'
    
    return iou_string


def get_iou(label, confusion_matrix):
    tp = confusion_matrix[label][label]

    # False negatives are everything in column (aside from true positives)
    fn = np.longlong(confusion_matrix[label,:].sum()) - tp

    # False positives are everything in row (aside from true positives)
    fp = np.longlong(confusion_matrix[:,label].sum()) - tp

    iou = tp / (tp + fp + fn)

    return iou

def get_file_id(file_basename, id_map:str):
    """Retrieves the dataset-specific ID of a file based on the ID map.
    id_map: A string containing the character '*'. Everything surrounding '*' will be removed from the filename to retreieve the ID."""
    # Sanity check
    if not '*' in id_map:
        raise ValueError("id_map must contain '*'")
    
    # Identify all elements to remove from id map
    to_remove = id_map.split(sep='*')
    file_id = file_basename

    # Remove all elements
    for remove_string in to_remove:
        file_id = file_id.replace(remove_string, '')

    return file_id

def _get_args():
    """Retrieve CLI args as dictionary"""
    parser = argparse.ArgumentParser(
        prog='evaluate_segmentation.py', description="Script evaluating performance of segmentation on any dataset."
    )

    parser.add_argument(
        '-d', '--dataset-info-path',
        help="""Path to JSON file defining dataset labels and ground_truth/prediction filenames. Format should be of form:
        ```
        {
            "groundTruthIdMap": <ground truth id map>,
            "predictionTruthIdMap": <prediction id map>,
            "labels": [
                {
                    "id": <id number>,
                    "name": <name string>,
                    "rgb", [<r>, <g>, <b>]
                }
            ],
        }
        ```
        'groundTruthIdMap' & 'predictionIdMap' should contain strings outlining the naming pattern for the ground truth & predictions respectively: Both must be strings defining the naming pattern, with the unique component defined by a '*'. 
        E.g. (for cityscapes ground truth): `"groundTruthIdMap": "*_gtFine_color.png"`
        """,
        type=str,
        required=True
    )

    parser.add_argument(
        '-t', '--ground-truth-glob',
        help="Glob string retrieving all ground truth values. E.g. (for cityscapes dataset): \"<...>/gtFine/val/**/*_gtFine_color.png\"",
        type=str,
        required=True
    )
    parser.add_argument(
        '-p', '--predictions-glob',
        help="Glob string retrieving all prediction values. Must return 1 image file for each ",
        type=str,
        required=True
    )

    args = parser.parse_known_args()[0]

    ## Sanity checking & reformatting
    args_dict = args.__dict__
    
    if not path.isfile(args_dict["dataset_info_path"]):
        raise ValueError("Dataset info path must be a file")
    
    with open(args_dict['dataset_info_path']) as f:
        dataset_info = json.load(f)

    args_dict['ground_truth_id_map'] = dataset_info['groundTruthIdMap']
    args_dict['prediction_id_map'] = dataset_info['predictionTruthIdMap']

    # Parse labels into label format
    labels = []
    for label in dataset_info['labels']:
        if len(label['rgb']) != 3:
            raise ValueError("Bad json file format")

        new_label = Label(
            id=label['id'],
            name=label['name'],
            rgb=label['rgb']
        )

        labels.append(new_label)

    args_dict['labels'] = labels

    return args_dict


def main():
    # Get CLI args
    args = _get_args()

    ## Generate confusion matrix
    confusion_matrix = create_confusion_matrix(args['labels'])

    ## Get prediction & ground truth lists
    args['ground_truth_glob'] = args['ground_truth_glob'].replace('\"', '') #FIXME remove
    args['predictions_glob'] = args['predictions_glob'].replace('\"', '')
    ground_truth_file_list = glob(args['ground_truth_glob'])
    prediction_file_list = glob(args['predictions_glob'])
    
    if not len(ground_truth_file_list) == len(prediction_file_list):
        raise ValueError(f"Number of ground truth ({len(ground_truth_file_list)}) and prediction files ({len(prediction_file_list)}) is not equal")

    ## Sort both lists (by image ID)
    # Generate lambdas to quickly retrieve ID of each image from its filename
    ground_truth_id_getter = lambda fpath: get_file_id(path.basename(fpath), args['ground_truth_id_map'])
    prediction_id_getter = lambda fpath: get_file_id(path.basename(fpath), args['prediction_id_map'])
    ground_truth_file_list.sort(key=ground_truth_id_getter)
    prediction_file_list.sort(key=prediction_id_getter)

    print("Starting...")
    loop_count = 1
    ## Perform pair-wise evaluation for each class
    for ground_truth_file, prediction_file in zip(ground_truth_file_list, prediction_file_list):
        # Check file IDs are the same
        if not(ground_truth_id_getter(ground_truth_file) == prediction_id_getter(prediction_file)):
            raise ValueError(f"Retrieved ground truth identifier ('{ground_truth_id_getter(ground_truth_file)}') and prediction identifier ('{prediction_id_getter(prediction_file)}') are not the same")

        print('\r' + f"Evaluating '{ground_truth_id_getter(ground_truth_file)}' ({loop_count}/{len(ground_truth_file_list)})", end='')
        loop_count += 1

        if loop_count > 10:
            break

        evaluate_pair(ground_truth_file, prediction_file, confusion_matrix, args['labels'])
    
    print("\nDone!")

    ## Display results
    print("Confusion Matrix:")
    print(display_confusion_matrix(confusion_matrix, args['labels']))

    print("IoUs:")
    print(display_ious(confusion_matrix, args['labels'])) 

if __name__ == '__main__':
    main()