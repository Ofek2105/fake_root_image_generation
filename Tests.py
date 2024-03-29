import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert into Grayscale
color_image = cv2.imread(r'res\type_3\arb_sr_x4.png')
# color_image = cv2.imread(r'res\type_2\zoomIn.png')
# color_image = cv2.imread(r'res\type_1\SR_P1_X4.png')

gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Filter (Unsharp Mask)
blurred_image = cv2.GaussianBlur(gray_image, (0, 0), 3)
unsharp_mask = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)

# Step 3: Convert the grayscale image back to a color image with 3 channels
color_image_gray = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)

# Step 4: Apply Mean Shift filtering on the color image
shifted_image = cv2.pyrMeanShiftFiltering(color_image_gray, 21, 51)

# Step 5: Apply Sobel Filter (Edge Detection)
sobel_x = cv2.Sobel(shifted_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(shifted_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Step 6: Convert the gradient magnitude image to a suitable data type for display
sobel_edges_display = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Step 7: Blend the images using the addWeighted function
blended_image = cv2.addWeighted(shifted_image, 1, sobel_edges_display, -0.5, 0)

# Compute the histogram of the shifted grayscale image
# histogram = cv2.calcHist([shifted_image], [0], None, [256], [0, 256])
#
# # Plot the histogram
# plt.figure()
# plt.plot(histogram, color='black')
# plt.xlabel('Grayscale Value')
# plt.ylabel('Magnitude')
# plt.title('Histogram of Shifted Grayscale Image')
# plt.xlim([0, 255])
# plt.grid(True)
# plt.show()

# Compute the connected components and labels
gray_shifted_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment regions
_, thresholded_image = cv2.threshold(gray_shifted_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Compute the connected components and labels on the thresholded image
_, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_image)

# Find the label of the brightest cluster (excluding background label 0)
brightest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

# Create a binary mask for the brightest cluster
brightest_mask = np.uint8(labels == brightest_label) * 255

# Display the image after edge detection
# cv2.imshow('Original Image', color_image)
# cv2.imshow('Sharpened Image', unsharp_mask)
# cv2.imshow('Edges after Sobel Filter', sobel_edges_display)
# cv2.imshow('Blended Image', blended_image)
# cv2.imshow('Mean Shift Filtered Image', shifted_image)
# cv2.imshow('Brightest Cluster Mask', brightest_mask)

cv2.imwrite(r'res\omer_tests\arb_sr_x4.png', brightest_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()