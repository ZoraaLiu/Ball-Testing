### Ball Detection and Diameter Calculation 
## This is a project that used in the industrial ball making process to detect the ball that is successfully made.
# This project is an implementation of a Python script that uses OpenCV library to catch images of balls and detect balls in images, enhance the contrast between the ball and the background, find its contour, calculate its diameter, and classify it into the value class of the ball close to the circle row.

## Requirement
To run this project, you need to have the following:
* Python 3.x installed
* OpenCV library installed (version 4.x or higher)
* NumPy library installed

## Usage
1. Clone or download the project from the GitHub repository to your local machine.
2. Open a command prompt or terminal window and navigate to the project directory.
3. Run the following command to execute the script:
" python ball_detection.py <image_file_path>"

Replace <image_file_path> with the path to the image file you want to analyze. The script will then perform the following steps:
Load the image and convert it to grayscale.
Enhance the contrast of the grayscale image.
Threshold the image to create a binary image.
Find the contours in the binary image using the RETR_EXTERNAL mode.
Calculate the diameter of the largest contour.
Classify the ball into the value class of the ball close to the circle row.

## Algorithm
The script uses the following algorithm to detect the ball and calculate its diameter:

1 Load the image and convert it to grayscale.
2 Enhance the contrast of the grayscale image using the CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm.
3 Threshold the image using Otsu's method to create a binary image.
4 Find the contours in the binary image using the RETR_EXTERNAL mode.
5 Filter the contours based on their area to remove noise and select the largest contour.
6 Fit a circle around the largest contour using the HoughCircles function.
7 Calculate the diameter of the circle.
8 Classify the ball into the value class of the ball close to the circle row based on its diameter.


