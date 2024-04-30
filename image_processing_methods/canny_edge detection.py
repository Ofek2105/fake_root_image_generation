import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

"""
This script takes an img and runs edge detection algorithms on this.
canny, sobel operation and hough transformation are performed.
"""


def sobel_edge_detection(image):
  """
  Performs Sobel edge detection on the given img.

  Args:
    image: A numpy array representing the img.

  Returns:
    A numpy array representing the edge detected img.
  """

  # Convert the img to grayscale if it is not already.
  if image.ndim == 3 and image.shape[2] == 3:
    image = np.mean(image, axis=2)

  # Create the Sobel kernels.
  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

  # Apply the Sobel kernels to the img.
  sobel_x_filtered = signal.convolve2d(image, sobel_x, mode='same')
  sobel_y_filtered = signal.convolve2d(image, sobel_y, mode='same')

  # Calculate the magnitude of the gradient.
  magnitude = np.sqrt(sobel_x_filtered ** 2 + sobel_y_filtered ** 2)

  # Normalize the magnitude to the range [0, 255].
  magnitude = (magnitude / np.max(magnitude)) * 255

  # Return the edge detected img.
  return magnitude.astype(np.uint8)


def canny_edge_detection(image, gawss_sigma):
  """
  Performs Canny edge detection on the given img.

  Args:
    image: A numpy array representing the img.

  Returns:
    A numpy array representing the edge detected img.
  """

  # Convert the img to grayscale if it is not already.
  if image.ndim == 3 and image.shape[2] == 3:
    image = np.mean(image, axis=2)

  # Apply Gaussian blur to reduce noise.
  blurred_image = cv2.GaussianBlur(image, (5, 5), gawss_sigma)

  # Convert the blurred img to a depth of CV_8U.
  blurred_image = cv2.convertScaleAbs(blurred_image)

  # Apply Canny edge detection.
  edges = cv2.Canny(blurred_image, 50, 150)

  # Return the edge detected img.
  return edges


def draw_lines(image, lines):
  """
  Draws the detected lines on the img.

  Args:
    image: A numpy array representing the img.
    lines: A list of tuples representing the lines detected in the img.
  """

  for line in lines:
    theta, x, y = line
    r = x * np.cos(theta) + y * np.sin(theta)
    cv2.line(image, (x, y), (int(r * np.cos(theta)), int(r * np.sin(theta))), (0, 0, 255), 2)


# image = cv2.imread(r'res\type_2\zoomIn.png')
image = cv2.imread(r'../res/type_3/arb_sr_x4.png')

# ========== canny - edge detection ==========
edge_detected_canny = canny_edge_detection(image, 1)
plt.title('canny - edge detection')
plt.imshow(edge_detected_canny, cmap='gray')
plt.show()


