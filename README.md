# **SAR Segmentation**

## **Introduction**

The pipeline offers a robust solution for handling image data of arbitrary sizes by employing a systematic approach of patching images into fixed-size patches. This not only enables efficient data augmentation but also facilitates training models on very high-resolution data that would otherwise exceed GPU memory limitations.

The process of training begins with the `train.py` script, where paths are created and essential utilities from `utils.py` are invoked. The data loading and preparation steps are handled in `dataset.py`, where the dataset is generated, split, and train, valid and test CSV files are saved. The image patching process, including saving co-ordinate of patches to train, valid, and json files is executed, followed by data augmentation and transformation through the `Augment` class. The dataset is then prepared using `MyDataset` class, which handles data fetching and transformation.

Metrics calculation, focal loss computation, and model initialization are handled in separate modules respectively `metrics.py`, `loss.py`, `model.py`, ensuring modularity and ease of management. Callback selection, including learning rate scheduling and validation visualization, is orchestrated through `SelectCallbacks` class.

For evaluation, two distinct scenario are provided in `test.py`. In the first scenario (evaluation = False), the evaluation dataset is prepared similarly to the training dataset, and predictions are made and displayed for analysis. In the second scenario (evaluation = True), additional steps for generating evaluation CSVs named eval_csv_gen is called

Throughout the pipeline, emphasis is placed on modularity, allowing for easy integration of new components or modifications. Understanding the intricacies of this pipeline is paramount for leveraging its capabilities effectively. For a deeper dive into its workings, referencing the FAPNET paper is recommended, providing insights into the architecture and rationale behind its design.

## **Dataset**

Different type of dataset ...............
1. ....
2. ....
3. ....

## **Model**

Our current pipeline supports semantic segmentation for both binary and multi-class tasks. To configure the pipeline for your desired number of classes, simply adjust the `num_classes` variable in the `config.py` file. No modifications are required in the `model.py` file. This straightforward configuration process allows for seamless adaptation to different classification requirements without the need for extensive code changes.

## **Setup**

First clone the github repo in your local or server machine by following:

```
git clone https://github.com/samiulengineer/road_segmentation.git
```

Change the working directory to project root directory. Use Conda/Pip to create a new environment and install dependency from `requirement.txt` file. The following command will install the packages according to the configuration file `requirement.txt`.

```
pip install -r requirements.txt
```

Keep the above mention dataset in the data folder that give you following structure. Please do not change the directory name `image` and `gt_image`.

```
--data
    --image
        --um_000000.png
        --um_000001.png
            ..
    --gt_image
        --um_road_000000.png
        --um_road_000002.png
            ..
```

## **Experiment**

* ### **Comprehensive Full Resolution (CFR)**:
This experiment utilize the dataset as it is. The image size must follow $2^n$ format like, $256*256$, $512*512$ etc. If we choose $300*300$, which is not $2^n$ format, this experiment will not work.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr \
    --weights False \
    --patchify False \
    --patch_class_balance False
```

* ### **Comprehensive Full Resolution with Class Balance (CFR-CB)**:
We balance the dataset biasness towards non-water class in this experiment.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment cfr_cb \
    --weights True \
    --balance_weights = include this in argparse \
    --patchify False \
    --patch_class_balance False
```

* ### **Patchify Half Resolution (PHR)**:
In this experiment we take all the patch images for each chip. Data preprocessing can handle any image shape and convert it to a specific patch size.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr \
    --weights False \
    --patchify False \
    --patch_class_balance False
```

* ### **Patchify Half Resolution with Class Balance (PHR-CB)**:
In this experiment we take a threshold value (19%) of water class and remove the patch images for each chip that are less than threshold value.

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --weights False \
    --patchify True \
    --patch_class_balance False \
```

* **Patchify Half Resolution with Class Balance Weight (PHR-CBW)**:

Double Class Balance + Ignore Boundary Layer .............
write something about it .......

```
python train.py --root_dir YOUR_ROOT_DIR \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --epochs 10 \
    --batch_size 3 \
    --index -1 \
    --experiment phr_cb \
    --weights True \
    --balance_weights = include this in argparse \
    --patchify False \
    --patch_class_balance False \
```

## Transfer Learning, Fine-tuning, and Training from Scratch

In this project, the behavior of the model training process is determined by certain variables in the `config.py` file. Here's how they influence the training approach:

### **Transfer Learning**

- If `transfer_lr` variable is set to `True` and a `load_model_name` is provided in `config.py`, the model will undergo transfer learning.

### **Fine Tuning**

- If `transfer_lr` is set to `False` but a `load_model_name` is provided in `config.py`, the model will undergo fine-tuning.

### **Training from Scratch**

- If `transfer_lr` is set to `False` and no `load_model_name` is provided in `config.py`, the model will be trained from scratch.



## Testing

* ### **CFR and CFR-CB Experiment**

Run following model for evaluating train model on test dataset.

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --index -1 \
    --patchify False \
    --patch_size 512 \
    --experiment cfr \
```

* ### **PHR and PHR-CB Experiment**

```
python test.py \
    --dataset_dir YOUR_ROOT_DIR/data/ \
    --model_name unet \
    --load_model_name my_model.hdf5 \
    --index -1 \
    --patchify True \
    --patch_size 256 \
    --experiment phr \
```

### **Evaluation from Image**

If you have the images without mask, we need to do data pre-preprocessing before passing to the model checkpoint. In that case, run the following command to evaluate the model without any mask.

1. You can check the prediction of test images inside the `logs > prediction > YOUR_MODELNAME > eval > experiment`.

```
python project/test.py \
    --dataset_dir YOUR_IMAGE_DIR/ \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```

### **Evaluation from Video**

Our model also can predict the road from video data. Run following command for evaluate the model on a video.

```
python project/test.py \
    --video_path PATH_TO_YOUR_VIDEO \
    --model_name fapnet \
    --load_model_name MODEL_CHECKPOINT_NAME \
    --experiment road_seg \
    --gpu YOUR_GPU_NUMBER \
    --evaluation True \
```





## **Action tree**


## Training Script: `train.py`

### Functions

- `create_paths(test=False)` - [utils.py]
- `get_train_val_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
     - `write_json()`
  - `Augment()` - Class
    - `call()`
      - `read_img()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `get_metrics()` - [metrics.py]
  - `MyMeanIOU()` - Class
- `focal_loss()` - [loss.py]
- `get_model_transfer_lr(model, num_classes)` - [model.py]
- `get_model()` - [model.py]
  - `models` - call all model functions
- `SelectCallbacks()` - Class - [utils.py]
  - `__init__()`
  - `lr_scheduler()`
  - `on_epoch_end()`
    - `val_show_predictions()`
      - `read_img()`
      - `transform_data()`
      - `display()`
  - `get_callbacks()`

## Test Script: `test.py`

### Evaluation = False

- `create_paths()`
- `get_test_dataloader()`
  - `data_csv_gen()`
    - `data_split()`
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display()`
- `get_metrics()`

### Evaluation = True

- `create_paths()`
- `get_test_dataloader()`
  - `eval_csv_gen()`*
    - `save_csv()`
  - `patch_images()`
    - `save_patch_idx()`
      - `class_percentage_check()`
    - `write_json()`
  - `MyDataset()` - Class
    - `__getitem__()`
      - `read_img()`
      - `transform_data()`
    - `get_random_data()`
      - `read_img()`
      - `transform_data()`
- `test_eval_show_predictions()`*
  - `read_img()`
  - `transform_data()`
  - `display_label()`*
  - `display()`
- `get_metrics()`
