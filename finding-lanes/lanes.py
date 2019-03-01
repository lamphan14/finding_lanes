import cv2 #help to load image
import numpy as np
import matplotlib.pyplot as plt #help with Region of interest
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
#create a function canny
def canny(image):
    #gradient: measure of change in brightness over adjacent pixels
    #strong gradient: 0 -> 255 indicates a steep change
    #Small gradient: 0 -> 15 indicate a shallow change
    #edge: rapid changes in brightness
    #each images are made up of pixels a 3-channel color image
    #each pixel in the image is a combination of 3 intensities: Red, Green, Blue
    #3 channels
    #Step2: convert to grayscale -> 1 channel
    #reasons: less computation, faster processing a single channel than a 3-channels color image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Step3:blur the image:
    #reasons: to reduce noises in the image and smoothen the image
    #applying Gaussian blur on a grayscale image with a 5x5 kernel
    blur = cv2.GaussianBlur(gray, (5,5),0) #reduce noises in grayscale image
    #step4: apply canny function:
    #reason: to identify edges in the image
    #an edge corresponds to a region in an image where there is a sharp change in intensity or color between adjacent pixels in the image
    # 0 0 225 225
    # 0 0 225 225
    # the change in brightness over a series of pixels is the gradients
    #we can represent an image in a two dimensional coordinate space X and Y
    #X: traverses image width //columns
    #Y: traverses image height //rows
    #the product of both width and height would give the total number of pixels in the image
    # with this method, we can:
    #1) look at the image as an array
    #2) as a continuous function of X and Y since it's a mathmematical function //f(x,y)
    #we can perform mathematical operation
    #question is: which operator can we use to determine a rapid changes in brightness for the image?
    #Canny function: derivative(f(x,y))
    #canny will perform a derivative on the function in both X and Y directions by measuring the change in intensity with respect to adjacent pixels
    #small derivative value = small change in intensity
    #big derivative value = big change in intensity
    # we are computing the gradients since the gradient is the change in brightness over a series of pixels
    # low threshold
    # high threshold
    # if the gradient is larger than the upper threshold then it is accepted
    # not accepted or rejected if below the lower threshold
    # if the gradient is between the thresholds then it will be accepted only if it is connected to a strong edge
    # the ratio of 1 -> 2 or 1 -> 3 is recommended by the documentation itself
    canny = cv2.Canny(blur, 50, 150) #outline the strongest gradient in the immage
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image
#step5: Region of interest
#takes in the image,
#return the enclosed of our field of view
#and recall tht enclosed region was triangular in shape
def region_of_interest(image):
    height = image.shape[0] #height now is 700
    polygons = np.array([
        [(200,height), (1100, height), (550, 250)]
         ])
    #specifiy the triable vertices that limiting the extent of our field of view we traced a triangle with vertices that go 200 along the X and vertically close to 700
    #second one being 1100 pixels along the X and Y is height
    mask = np.zeros_like(image)
    #fill the mask with the triangle
    cv2.fillPoly(mask, polygons, 255)
    #computing the bitwise & of both image
    #take the bitwise & of each homologous pixel in both arrays
    #ultimately masing the canny image to only show the region of interest traced by the polygonal contour of the mask.
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
#step1: load the image
#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image,lines)
#line_image = display_lines(lane_image, averaged_lines)
#switch to normal pic with the highlight
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('result',combo_image)
#cv2.waitKey(0) #display the image for a specified amount of 0 millisecond
#plt.imshow(canny)
#plt.show()
#step6: finding lane lines //bitwise_and
#step final: finding lanes on video frame:
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read(())
    #copy the vode from lines 102 to 110 since the same thing would be applied
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame, averaged_lines)
    #switch to normal pic with the highlight
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    #display the image for a specified amount of 1 millisecond
    #when we hit "Q" button will stop the program
        break
cap.release()
cv2.destroyAllWindows()
