import numpy as np
import cv2
import rdp


def get_polygons_bbox_from_bin_image(bin_image, neg_mask=None):
  result = bin_image.copy()
  if neg_mask is not None:
    result = np.logical_and(bin_image.copy(), np.logical_not(neg_mask.copy()))

  # Dilate the binary image to thicken thin objects
  kernel = np.ones((3, 3), np.uint8)
  dilated_result = cv2.dilate(result.astype(np.uint8), kernel, iterations=1)

  # Find contours using OpenCV
  contours, _ = cv2.findContours(dilated_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    max_contour = max(contours, key=cv2.contourArea)
    contour_points = np.squeeze(max_contour).astype(np.float32)

    if contour_points.ndim == 1:  # Handle the case where contour is a single point
      contour_points = np.expand_dims(contour_points, axis=0)

    contour_points[:, 0] = contour_points[:, 0] / bin_image.shape[0]
    contour_points[:, 1] = contour_points[:, 1] / bin_image.shape[1]

    # Simplify the contour using RDP
    keep = rdp.rdp(contour_points.tolist(), epsilon=1e-3, algo="iter", return_mask=True)
    polygon_points = contour_points[keep].flatten().tolist()

    # Calculate bounding box
    min_x, min_y = np.min(contour_points, axis=0)
    max_x, max_y = np.max(contour_points, axis=0)
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

    return polygon_points, bbox
  return [], []
