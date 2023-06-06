import cv2
import numpy as np

class BallAnalyzer:
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.processed_image = None
        self.contours = None
        self.circles = None
        self.histogram = None
        self.circle_area = None
        self.percentage_error = None
    
    def loadImage(self, image_path): 
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def enhanceContrast(self):
        # Apply adaptive histogram equalization (CLAHE) for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gray_image = clahe.apply(self.gray_image)

    def preprocessImage(self):
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        # Apply adaptive thresholding to obtain a binary image
        _, self.processed_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def morphology(self):
        # Apply erosion and closing operations multiple times for denoising
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(2):
            self.processed_image = cv2.erode(self.processed_image, kernel)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)


    #def morphologicalGradient(self):
        # Apply morphological gradient to emphasize edges of objects
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #gradient_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_GRADIENT, kernel)
        #_, self.processed_image = cv2.threshold(gradient_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def detectContours(self):
        # Find contours in the processed image
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def fitCircles(self):
        self.circles = []
        for contour in self.contours:
            # Fit a circle to each contour
            if len(contour) >= 5: # if sufficient number of points
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                self.circles.append((center, radius))

    def ModifiedImage(self):
        if self.image is not None:
            modified_image = self.image.copy()

            for contour in self.contours:
                if len(contour) >= 5:  # if sufficient number of points
                    # Split the contour into line segments
                    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                    # Fit a circle to each line segment
                    for i in range(len(approx) - 1):
                        # Get the start and end points of the line segment
                        p1 = tuple(approx[i][0])
                        p2 = tuple(approx[i + 1][0])

                        # Calculate the center and radius of the circle
                        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                        radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) // 2)

                        # Draw the circle on the modified image in red color
                        cv2.circle(modified_image, center, radius, (0, 0, 255), thickness=2)

                    # Draw a circle between the last and first points to close the contour
                    p1 = tuple(approx[-1][0])
                    p2 = tuple(approx[0][0])
                    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) // 2)
                    cv2.circle(modified_image, center, radius, (0, 0, 255), thickness=2)


    def computeHistogram(self):
        # Extract the radii values from the detected circles
        radii = [radius for (_, radius) in self.circles]
        self.histogram, _ = np.histogram(radii, bins=10, range=(0, np.max(radii)))

    def calculateCircleArea(self):
        # Calculate the area of the largest circle
        if self.circles:
            # Find the largest circle based on its radius
            _, max_radius = max(self.circles, key=lambda x: x[1])
        
            # Calculate the area of the circle 
            self.circle_area = np.pi * max_radius**2

    def calculatePercentageError(self):
        # Calculate the percentage error of circle area compared to the original shape
        original_area = self.image.shape[0] * self.image.shape[1]
        self.percentage_error = abs((self.circle_area - original_area) / original_area) * 100

    def analyzeImage(self, image_path):
        self.loadImage(image_path)
        self.enhanceContrast()
        self.preprocessImage()
        self.morphology()
        #self.morphologicalGradient()
        self.detectContours()
        self.fitCircles()
        self.computeHistogram()
        self.ModifiedImage()
        self.calculateCircleArea()
        self.calculatePercentageError()

    def printResults(self):
        if self.circles:
            print("Detected Circles:")
            for idx, (center, radius) in enumerate(self.circles):
                print(f"Circle {idx+1}: Center={center}, Radius={radius}")
        else:
            print("No circles detected.")

        print("Histogram of Radii:")
        print(self.histogram)

        cv2.imshow("Processed Image", self.processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()

    
#analyzer = BallAnalyzer()
#analyzer.analyzeImage('/Users/liushixiao/Downloads/WechatIMG57_page-0001.jpg')
#analyzer.printResults()


