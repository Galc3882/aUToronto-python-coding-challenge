import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def getImages():
    """
    Get images from the input folder
    Creates a generator to iterate over the images
    Sends the image and the name of the image as a tuple
    """
    # if folder does not exist, return
    if not os.path.exists(r"./input"):
        return

    for img in os.listdir(r"./input"):
        # changes the image from BGR to RGB
        yield (cv2.cvtColor(cv2.imread(r"./input/"+img), cv2.COLOR_BGR2RGB), img)


class SegmentationMask:
    def __init__(self):
        self.mask = None # mask for the image as a numpy matrix
        self.stack = [] # contour stack

        # boundaries for orange color range values
        self.lower = np.array([150, 100, 20])
        self.upper = np.array([180, 255, 255])

    def maskify(self, img):
        """
        create a mask for the image
        """
        # RGB to HSV
        HSVImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
        # create a mask for the orange color
        self.mask = cv2.inRange(HSVImg, self.lower, self.upper)

        # find all the contours in the mask
        contour, _ = cv2.findContours(
            self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # get the biggest contour size
        biggest_area = -1
        for con in contour:
            area = cv2.contourArea(con)
            if biggest_area < area:
                biggest_area = area

        # add contours that are at least 30% of the biggest contour
        self.stack = []
        for con in contour:
            if cv2.contourArea(con) > 0.3 * biggest_area:
                self.stack.append(con)

        # reset the mask
        self.mask = np.zeros(self.mask.shape, np.uint8)
        # draw the contours on the mask
        self.mask = cv2.drawContours(self.mask, self.stack, -1, 255, -1)
        # add some padding to the mask
        self.mask = cv2.dilate(self.mask, np.ones(
            (5, 5), np.uint8), iterations=1)
        # binary to RGB
        self.mask = np.array([[np.zeros(3) if self.mask[i][j] != 255 else np.array([255, 255, 255])
                               for j in range(len(self.mask[0]))] for i in range(len(self.mask))])

    def ShowImage(self, img, mask):
        """
        show the images
        """
        fig, axs = plt.subplots(1, 2)
        self.axs = axs
        axs[0].axis("off")
        axs[0].imshow(mask)
        axs[1].imshow(img)
        axs[1].axis("off")
        plt.show()


def main():
    """
    Main function
    Iterates over the images in the input folder
    and creates a mask for each image,
    then stores the mask in the output folder
    """
    # if output folder does not exist, create it
    if not os.path.exists(r"./output"):
        os.mkdir(r"./output")
    # create a SegmentationMask object
    seg = SegmentationMask()
    for img in getImages():
        # label the image with the closest color
        seg.maskify(img[0])
        
        # seg.ShowImage(img[0], seg.mask)

        # save image to output folder
        cv2.imwrite(r"./output/"+img[1], seg.mask)


if __name__ == "__main__":
    main()
