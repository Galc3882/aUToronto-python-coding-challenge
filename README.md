# aUToronto python coding challenge
# What is it supposed to do?

fsdfsff

## Example:

# Requirements and file architecture:

# How to run? 

# Limitations and known bugs:

## next steps:

# How does it work? 

This does... using...

Each function works like this...

# Process:

I started out by labeling each pixel that was within some range of orange. OpenCV inverts the red and blue colors so when viewing the pictures, the orange will look blue.
### The result looked like this:

![Figure_1](https://user-images.githubusercontent.com/86870298/180009708-674b157b-6bff-46fd-a72f-a4315419422c.png)

I the chucked it into a kmeans algorithm in hopes that it would make the mask better and select a more optimized centroid. After some adjustments and playing around, I was not able to get the result I wanted.
After only one iteration, the centriods moved from  

0: [220, 50, 50]

1: [0  , 0 , 0 ]

to 

0: [91.80638, 72.726364, 68.61953]

1: [211.5603, 205.13245, 194.565 ]


### And the result was:

![Figure_1](https://user-images.githubusercontent.com/86870298/180011145-f3e6743f-1678-4797-b668-d43d794bc5ab.png)


The kmeans changes the centroids too much to decrease the error and that clearly was not good for what I was trying to achieve. I then thought about using knearest neighbors but after looking at the implementation, I decided on building my own custom function.

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
