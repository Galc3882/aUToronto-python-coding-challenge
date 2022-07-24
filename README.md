# aUToronto python coding challenge
# What is it supposed to do?

segmentation_mask.py outputs the segmentation mask of orange barrels in pictures. 

## Example:

![2_3 1](https://user-images.githubusercontent.com/86870298/180622921-41e9b082-9fb9-4ad9-a46e-7acd7e16bcc7.png)

![2_3 1](https://user-images.githubusercontent.com/86870298/180622917-3c760cf3-8880-4fa4-8302-06759e7cbaab.png)

# Requirements and file architecture:
### Imports:
```python
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
```
## Instructions:
- Put all the pictures in the input folder. 
- Run segmentation_mask.py.
- Your segmentation masks will be in the folder named output.

# Limitations and known bugs:
While the performance for the segmentation mask is pretty good, the program does have to switch from using RGB to HSV which is something to investigate. OpenCV loads pictures as BGR, so I also turn them into RGB to show the pictures in matplotlib. As well, going back and forth between grayscale and RGB can be time consuming.

There are a few known bugs with the program, the main one is that it picks out the biggest blob of orange, then saves all the blobs that are at least 30% as big as it. That means that if there is another orange item in the picture that in bigger or at least 30% as big as the barrels, it will be included. Additionally, if a similar orange color on close (behind or in front) to the barrel, they will be considered as the same blob of color.
If the lighting is bad, parts of the orange color might not be within range of the program and only part of the barrel will be in the segmentation mask. 

## Example of bad detection:

![Figure_1](https://user-images.githubusercontent.com/86870298/180654217-2bb1df07-e74a-4928-ba03-c724295112df.png)

![Figure_2](https://user-images.githubusercontent.com/86870298/180654218-7e1730de-2f17-42c0-835b-9618750affe2.png)

## Next steps:
I would:
- investigate fixing the known bugs.
- clean the mask outline.
- Use shape detection to understand which blob is the barrel.

# How does it work? 
The program iterates through each image in the input folder. It then casts the picture into HSV color space and transforms it into a binary matrix based on if the pixel is an orange color. After finding the main blobs of color, it dilates the m by five pixels to clean the edges and turns it back into RGB for the PNG output. It then saves the image with the same name as the input.

The ```ShowImage ``` function show a side-by-side view of the image and the segmentation mask.

# Process:

I started out by labeling each pixel that was within some range of orange.
### The result looked like this:

![Figure_1](https://user-images.githubusercontent.com/86870298/180024619-f1637cb9-991a-4f75-be4c-03f0fdb709d1.png)


I the chucked it into a kmeans algorithm in hopes that it would make the mask better and select a more optimized centroid. After some adjustments and playing around, I was not able to get the result I wanted.
After only one iteration, the centroids moved from  

```python
0: [220, 50 , 50 ]
1: [0  , 0  , 0  ]
```
to 
```python
0: [91 , 72 , 68 ]
1: [211, 205, 194]
```


### And the result was:

![Figure_1](https://user-images.githubusercontent.com/86870298/180024871-bc7b2d7d-fe66-4ea0-aa9c-af55f720c4af.png)


The kmeans changes the centroids too much to decrease the error and that clearly was not good for what I was trying to achieve. I then thought about using k-nearest neighbors but after looking at the implementation, I decided looking into OpenCV’s computer vision library. I found two functions that were instrumental in my implementation.

```python
cv2.inRange
```
This function receives a higher and lower bound for HSV color space and returns a binary matrix of if a pixel is within that range. 
### And the result was:

![image](https://user-images.githubusercontent.com/86870298/180622652-0ec4480c-ce86-4f52-98cd-15ee7823095a.png)

```python
cv2.findContours
```
This function takes a grayscale matrix and returns a list of all the outlines of the shapes in the image.
Using these functions, I was able to divide the picture into blobs of orange color and pick out the biggest ones.
### And the result was:

![Figure_4](https://user-images.githubusercontent.com/86870298/180622786-a1fbd74a-031c-4249-a3ed-206a47324d1e.png)

