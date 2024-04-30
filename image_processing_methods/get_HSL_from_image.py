import cv2
import numpy as np


# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    # Get the HSV values of the clicked pixel
    pixel_hsv = hsv_image[y, x]
    # print(f"HSV Values at ({x}, {y}): Hue={pixel_hsv[0]}, Saturation={pixel_hsv[1]}, Value={pixel_hsv[2]} => ")
    print(f"({pixel_hsv[0]}, {pixel_hsv[1]}, {pixel_hsv[2]}), ", end="")


# Load the img
image = cv2.imread('SR_P1_X4.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window and set the mouse callback function
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)

# Display the img until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
