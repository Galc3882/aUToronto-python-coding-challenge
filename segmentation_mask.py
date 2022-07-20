import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def getImages():
        """
        Get images from the input folder
        """
        for img in os.listdir(r"./input"):
            yield cv2.cvtColor(cv2.imread(r"./input/"+img), cv2.COLOR_BGR2HSV)

class SegmentationMask:
    def __init__(self):
        self.orange = np.array([200, 20, 30])  # Orange color IN RGB
        self.PIC_REDUCTION_FACTOR = 1/8  # reduce the size of the image to speed up the process
        self.COLOR_THRESHOLD = 500  # threshold for the distance between colors
        self.PIXEL_THRESHOLD =  8 # threshold for the number of pixels that are close to the given color

    def preprocess(self, img):
        """
        Preprocess the image by changing the color space to RGB and reshaping the image to 3 arrays
        """
        return np.float32(img.reshape((-1, 3)))


    def labelInialCentroids(self, img):
        """
        Label the initial centroids with the closest colors to the given color
        """
        # # creat a numpy array with the same size as the image and set all the values to 0
        # labels = np.zeros(
        #     len(img)*len(img[0]), dtype="uint8").reshape(len(img), len(img[0]))
        # # loop through the image and put 255 in the labels array if the color is close to the given color
        # # else increase the threshold and try again
        # selection = 0 
        # colorThreshold = 0
        # while selection < self.PIXEL_THRESHOLD:
        #     colorThreshold += self.COLOR_THRESHOLD
        #     for i in range(len(img)):
        #         for j in range(len(img[0])):
        #             if self.isOrange(img[i][j]):
        #                 labels[i][j] = 255
        #                 selection += 1
        # return labels

        
        # define range of red color in HSV
        lower_red = np.array([160,50,50])
        upper_red = np.array([180,255,255])
            
        # Threshold the HSV image using inRange function to get only red colors
        mask = cv2.inRange(img, lower_red, upper_red)
        plt.imshow(mask)


    def isOrange(self, color):
        """
        calculate the distance between two colors
        """
        return (color[0]>345 or color[0] <60) and (color[1]!=0) and (color[1]>10)


    def ShowImage(self, reducedImg, labels):
        """
        show the images
        """
        reducedImg = cv2.cvtColor(reducedImg, cv2.COLOR_HSV2RGB)

        labels = labels.reshape(len(reducedImg), len(reducedImg[0]))
        labels = [[[0, 0, 0] if labels[i][j] == 0 else reducedImg[i][j] if labels[i][j] == 255 else [0, 0, 100]
                for j in range(len(labels[0]))] for i in range(len(labels))]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(labels)
        axs[0].axis("off")
        axs[1].imshow(reducedImg)
        axs[1].axis("off")

        # plt.savefig("test.png", bbox_inches='tight')
        plt.show()


    def findLargestCluster(self, labels):
        """
        find the index of the largest cluster of pixels 
        by finding the longest row and column of pixels that equal 255
        """
        sumRows = np.sum(labels, axis=1)
        i = np.argmax(sumRows)
        sumColums = np.sum(labels, axis=0)
        j = np.argmax(sumColums)
        return i, j


    def indexOfBarrel(self, img):
        """
        Find the index of the barrel in the image
        """
        # reduce the size of the image to speed up the process
        reducedImg = cv2.resize(img, dsize=(
            int(len(img[0])*self.PIC_REDUCTION_FACTOR), int(len(img)*self.PIC_REDUCTION_FACTOR)), interpolation=cv2.INTER_CUBIC)
        # !adjust the color to the closest color from the image to the given color
        # !color = self.findMostSimilarColor(reducedImg)
        # label the image with the closest color
        label = self.labelInialCentroids(reducedImg)
        # find the indecies of the centroid of the largest cluster
        i, j = self.findLargestCluster(label)


        label[i][j] = 1

        # !show images
        self.ShowImage(reducedImg, label)
        return label
        
    def findMostSimilarColor(self, img):
        """
        find the most similar color to the given color by averaging the some of the similar colors in the image

        loop through the image and add the colors to the array
        if the color is close to the given color
        then return the average of the colors
        else increase the threshold and try again
        """
        colorThreshold = 0
        while True:
            colorThreshold += self.COLOR_THRESHOLD
            print(colorThreshold)
            selection = []
            for i in range(len(img)):
                for j in range(len(img[0])):
                    t = self.distanceBetweenColors(img[i][j], self.orange)
                    if t < colorThreshold:
                        selection.append(img[i][j].tolist())
            if len(selection) > self.PIXEL_THRESHOLD:
                closestColor = np.mean(selection, axis=0).astype(int)
                break
        return closestColor

def main():
    seg = SegmentationMask()
    for img in getImages():
        labels = seg.indexOfBarrel(img)
        # plt.imshow(labels, cmap="gray")
        # plt.show()


if __name__ == "__main__":
    main()
