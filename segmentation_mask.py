import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def getImages():
    """
    Get images from the input folder
    """
    for img in os.listdir(r"./input"):
        yield cv2.imread(r"./input/"+img)


def preprocess(img):
    """
    Preprocess the image by changing the color space to RGB and reshaping the image to 3 arrays
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.float32(img.reshape((-1, 3)))


def labelInialCentroids(color, img):
    """
    Label the initial centroids with the closest colors to the given color
    """
    # creat a numpy array with the same size as the image and set all the values to 0
    labels = np.zeros(len(img), dtype="uint8")
    # loop through the image and put 255 in the labels array if the color is close to the given color
    for i in range(len(img)):
        t = distanceBetweenColors(color, img[i])
        if t < 5000:
            labels[i] = 255
    return labels


def distanceBetweenColors(color1, color2):
    """
    calculate the distance between two colors
    """
    return (color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2


def ShowImage(reducedImg, labels):
    labels = labels.reshape(len(reducedImg), len(reducedImg[0]))
    labels = [[[0, 0, 0] if labels[i][j] == 0 else reducedImg[i][j]
               for j in range(len(labels[0]))] for i in range(len(labels))]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(labels)
    axs[1].imshow(reducedImg)
    plt.show()


def kMeans(img, maxIterations, centroid):
    """
    Perform k-means clustering on the image
    """

    # reduce the size of the image to speed up the process
    reducedImg = cv2.resize(img, dsize=(
        int(len(img[0])/16), int(len(img)/16)), interpolation=cv2.INTER_CUBIC)
    # preprocess the reduced image
    reducedPredImg = preprocess(reducedImg)
    # process the actual image for kmeans
    predImg = preprocess(img)
    # label the image with the closest color
    labels = labelInialCentroids(centroid, reducedPredImg)
    # !show images
    # ShowImage(reducedImg, labels)
    # preform k-means clustering
    ret, newLabel, center = cv2.kmeans(
        predImg, 2, bestLabels=labels, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxIterations, 255.0), attempts=10, flags=cv2.KMEANS_USE_INITIAL_LABELS, centers=np.array([centroid, [0, 0, 0]]))
    # !show the new image
    ShowImage(img, newLabel)
    return newLabel


def main():
    Orange = np.array([220, 50, 50])  # Orange color IN RGB
    for img in getImages():
        labels = kMeans(img, 1, Orange)
        labels = labels.reshape((img.shape[0], img.shape[1]))
        plt.imshow(labels, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
