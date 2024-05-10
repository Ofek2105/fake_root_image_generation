import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.special import binom


def bezier_curve(points, n=100):
  """Generates n points on a Bezier curve defined by control points."""
  t = np.linspace(0, 1, n)
  curve = np.zeros((n, 2))
  N = len(points) - 1
  for i, point in enumerate(points):
    binom_coeff = binom(N, i)
    term = binom_coeff * (t ** i) * ((1 - t) ** (N - i))
    curve += np.outer(term, point)
  return curve


def draw_my_thick_bezier_curve(bin_image, s1_point, s2_direction, m_point, e_point, thickness, weight=3):
  points = [s1_point, m_point, e_point]
  curve = bezier_curve(points, weight * 100)  # weight controls the resolution of the curve

  # Calculate tapering thickness along the curve, avoid going to zero
  t = np.linspace(0, 1, len(curve))
  min_thickness = 1  # Maintain a minimum thickness to avoid discontinuity
  thickness_profile = thickness * (1 - t ** 2) + min_thickness  # Quadratic tapering plus minimum thickness

  # Calculate normals for the thickness
  dx = np.gradient(curve[:, 0])
  dy = np.gradient(curve[:, 1])
  normals = np.column_stack([-dy, dx])
  norm_lengths = np.linalg.norm(normals, axis=1)
  normals /= norm_lengths[:, np.newaxis]
  normals *= thickness_profile[:, np.newaxis]  # Apply tapered thickness

  # Create the upper and lower borders of the "hair"
  upper_curve = curve + normals
  lower_curve = curve - normals

  # Fill polygon on binary image
  cc, rr = polygon(np.append(upper_curve[:, 1], lower_curve[::-1, 1]),
                   np.append(upper_curve[:, 0], lower_curve[::-1, 0]))
  valid = (rr >= 0) & (rr < bin_image.shape[0]) & (cc >= 0) & (cc < bin_image.shape[1])
  bin_image[rr[valid], cc[valid]] = 1  # Set pixels within polygon to 1

  # Calculate bounding box
  min_row, max_row = np.min(rr[valid]), np.max(rr[valid])
  min_col, max_col = np.min(cc[valid]), np.max(cc[valid])
  bbox = [min_col, min_row, max_col - min_col, max_row - min_row]

  # Construct polygon points
  polygon_points = [coord for pair in zip(cc[valid], rr[valid]) for coord in pair]

  # # Optionally, plot for visualization
  # plt.imshow(bin_image, cmap='gray')
  # plt.plot(upper_curve[:, 0], upper_curve[:, 1], 'b--')
  # plt.plot(lower_curve[:, 0], lower_curve[:, 1], 'g--')
  # plt.fill_betweenx(curve[:, 1], upper_curve[:, 0], lower_curve[:, 0], color='red', alpha=0.3)
  # plt.title("Smooth Tapered Hair-like Bezier Curve on Binary Image")
  # plt.show()

  return polygon_points, bbox


# Example usage:
img_height, img_width = 300, 300
bin_image = np.zeros((img_height, img_width), dtype=bool)
s1_point = (10, 10)
s2_direction = (50, 0)  # This example does not use s2_direction in computation
m_point = (10, 60)
e_point = (40, 100)
thickness = 5

points, bbox = draw_my_thick_bezier_curve(bin_image, s1_point, s2_direction, m_point, e_point, thickness)
