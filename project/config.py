from datetime import date
import json
from pathlib import Path


# Image Input/Output
# ----------------------------------------------------------------------------------------------
channel_type = ["vv","vh","nasadem"] # we need to pass number of channels as list
channel_type = ["vv","vh","nasadem"] # we need to pass number of channels as list
in_channels = len(channel_type)
num_classes = 2
height = 400 # for PHR-CB experiment patch size = height = width
width = 400


# Training
# ----------------------------------------------------------------------------------------------
model_name = "fapnet"
batch_size = 8
epochs = 30
learning_rate = 3e-4
val_plot_epoch = 3
augment = False
transfer_lr = False
gpu = "0"

# Dataset
# --------------------------------mask--------------------------------------------------------------
weights = False # False if cfr, True if cfr_cb
balance_weights = [7, 93]
root_dir = Path("/mnt/hdd2/mdsamiul/project/imseg_sar_csml")
dataset_dir = root_dir / "data/dataset/"
train_size = 0.8
train_dir = root_dir / "data/csv/train.csv"
valid_dir = root_dir / "data/csv/valid.csv"
test_dir = root_dir / "data/csv/test.csv"
eval_dir = root_dir / "data/csv/eval.csv"

# Patchify (phr & phr_cb experiment)
# ----------------------------------------------------------------------------------------------
patchify = True
patch_class_balance = True # whether to use class balance while doing patchify
patch_size = 256 # height = width, anyone is suitable
stride = 128

p_train_dir = root_dir / f"data/json/train_patch_phr_cb_{patch_size}_{stride}.json"
p_valid_dir = root_dir / f"data/json/valid_patch_phr_cb_{patch_size}_{stride}.json"
p_test_dir = root_dir / f"data/json/test_patch_phr_cb_{patch_size}_{stride}.json"
p_eval_dir = root_dir / f"data/json/eval_patch_phr_cb_{patch_size}_{stride}.json"

# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
csv = True # required for csv logger
val_pred_plot = True
lr = True
tensorboard = True
early_stop = False
checkpoint = True
patience = 300 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

# Evaluation
# ----------------------------------------------------------------------------------------------
load_model_name = 'fapnet_ex_patchify_WOC_256_epochs_2000_12-Apr-22.hdf5'
load_model_dir = None #  If None, then by befault root_dir/model/model_name/load_model_name
evaluation = True # default evaluation value will not work
video_path = None

# Prediction Plot
# ----------------------------------------------------------------------------------------------
index = -1 # by default -1 means random image else specific index image provide by user

#  Create config path
# ----------------------------------------------------------------------------------------------
# do not need this condition
if patchify:
    height = patch_size
    width = patch_size
    
# Experiment Setup
# ----------------------------------------------------------------------------------------------
# cfr, cfr-cb, phr, phr-cb, phr-cbw
experiment = f"{str(date.today())}_e_{epochs}_p_{patch_size}_s_{stride}"

# Create Callbacks paths
# ----------------------------------------------------------------------------------------------
tensorboard_log_name = "{}_ex_{}_ep_{}".format(model_name, experiment, epochs)
tensorboard_log_dir = root_dir / "logs/tens_logger" / model_name

csv_log_name = "{}_ex_{}_ep_{}.csv".format(model_name, experiment, epochs)
csv_log_dir = root_dir / "logs/csv_logger" / model_name   
csv_logger_path = root_dir / "logs/csv_logger"

checkpoint_name = "{}_ex_{}_ep_{}.hdf5".format(model_name, experiment, epochs)
checkpoint_dir = root_dir / "logs/model" / model_name

# Create save model directory
# ----------------------------------------------------------------------------------------------
if load_model_dir == None:
    load_model_dir = root_dir / "logs/model" / model_name
    
# Create Evaluation directory
# ----------------------------------------------------------------------------------------------
prediction_test_dir = root_dir / "logs/prediction" / model_name / "test" / experiment
prediction_eval_dir = root_dir / "logs/prediction" / model_name / "eval" / experiment
prediction_val_dir = root_dir / "logs/prediction" / model_name / "validation" / experiment

# Create Visualization directory
# ----------------------------------------------------------------------------------------------
visualization_dir = root_dir / "logs/visualization"
