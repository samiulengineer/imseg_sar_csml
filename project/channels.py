import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

eval_csv = pd.read_csv("/mnt/hdd2/mdsamiul/project/imseg_csml/data/csv/train.csv")
masks = eval_csv["feature_ids"].to_list()
ext = ["_vv.tif","_vh.tif","_nasadem.tif"]
masks = masks[0]
masks= [masks+ex for ex in ext]
print(masks)
for p in masks:
    with rasterio.open(p) as im:
        image = im.read(1)
        print("...............................")
        # print(p)
        # print(np.unique(image, return_counts=False))
        print(np.mean(image))
        print(np.std(image))
        print("...............................")

# print(masks)
# with rasterio.open("/mnt/hdd2/mdsamiul/project/dataset/rice_field_segmentation/id_1/w1000_h1000/data2/patch_0_0_w1000_h1000_id_1.tif") as im:
#     image = im.read(1)
# print(image.shape)
# print(len(image.shape))
# print(f"unique value of mask {np.unique(image)}")
# plt.imshow(image)
# print(type(image))
# # print(image)
# plt.imshow(image, cmap='viridis')  # You can choose a different colormap if needed
# plt.axis('off')  # Turn off axis labels and ticks

# # Save the figure as a PNG image
# plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)

# Show the plot (optional)
# plt.show()

# mask = cv2.imread("/mnt/hdd2/mdsamiul/project/dataset/rice_field_segmentation/id_1/w1000_h1000/data2/patch_0_0_w1000_h1000_id_1.tif")
# print(mask.shape)
# print(len(mask.shape))


# from PIL import Image

# # Load the TIFF image
# img = Image.open("/mnt/hdd2/mdsamiul/project/dataset/rice_field_segmentation/id_1/w1000_h1000/data2/patch_0_0_w1000_h1000_id_1.tif")

# # Print the shape
# print(img.size)
