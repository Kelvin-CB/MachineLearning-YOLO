
BCCD Augmentation - v1 2022-04-01 11:38pm
==============================

This dataset was exported via roboflow.ai on April 2, 2022 at 2:41 AM GMT

It includes 840 images.
Blood-Cells are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random rotation of between -30 and +30 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically


