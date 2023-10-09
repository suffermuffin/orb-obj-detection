# orb-obj-detection

This project was an attempt to get bounding box from image without training
using [ORB](https://github.com/orb-community/orb) with sample images. The idea is that with a small number of samples of searched
object it is possible to create ok-ish bounding box with simillar features using [ORB](https://github.com/orb-community/orb).  For example, in this case banana dataset was
used. To create bounding boxes user must set a number of images from specified directory
in a `range()`-like manner. Then specifie the number of images where the images in which 
to create bounding boxes. Then use `.fit` method to creat keypoints and Hamming distances
for each image. Last step is to choose alghoritm to create bounding boxes, which are:
`.get_abox_median()`, `.get_aboxes_dispersion()` and `.get_aboxes_non_min_sup()` for now. They are 
described in their DOCstring. The last thing user can do is to compare their IoU metric with `.get_iou()`
method.

The result is quite unsetteling, as IoU metric won't overcome the 25% accuracy mark in the most tests. And the best one
comes from `.get_aboxes_non_min_sup()` which is I belive is due to dataset's syntetic nature.