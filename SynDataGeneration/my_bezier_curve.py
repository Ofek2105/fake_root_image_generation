import numpy as np
from image_processing_methods.IP_funcs import get_polygons_bbox_from_bin_image
from skimage.draw import polygon
from scipy.special import binom
import matplotlib.pyplot as plt
import cv2


def bezier_curve(points, n=100):
    """Generates n points on a Bézier curve defined by control points."""
    t = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))
    N = len(points) - 1
    for i, point in enumerate(points):
        binom_coeff = binom(N, i)
        term = binom_coeff * (t ** i) * ((1 - t) ** (N - i))
        curve += np.outer(term, point)
    return curve


def draw_my_thick_bezier_curve(bin_image,
                               s1_point,
                               m_point,
                               e_point,
                               thickness,
                               weight=2,
                               neg_mask=None):
    weight = 2 if weight == 0 else 100

    points = [s1_point, m_point, e_point]
    curve = bezier_curve(points, weight)  # weight controls the resolution of the curve

    t = np.linspace(0, 1, len(curve))
    min_thickness = 0.6  # Maintain a minimum thickness_ to avoid discontinuity
    # thickness_profile = thickness * (1 - t ** 2) + min_thickness  # Quadratic tapering plus minimum thickness_
    # min_thickness = 0.5  # Minimum practical thickness
    thickness_profile = np.maximum(min_thickness, thickness * np.exp(-5 * t ** 2))

    # Calculate normals for the thickness_
    lower_curve, upper_curve = get_upper_lower_curve(curve, thickness_profile)

    # Fill polygon on binary image
    cc, rr = polygon(np.append(upper_curve[:, 0], lower_curve[::-1, 0]),  # x-coordinates
                     np.append(upper_curve[:, 1], lower_curve[::-1, 1]))  # y-coordinates
    if len(cc) == 0:
        return [], []

    empty_bin = np.zeros_like(bin_image)
    valid = (rr >= 0) & (rr < empty_bin.shape[0]) & (cc >= 0) & (cc < empty_bin.shape[1])
    empty_bin[rr[valid], cc[valid]] = 1  # draw the hair on empty for annotation
    polygon_points, bbox = get_polygons_bbox_from_bin_image(empty_bin, neg_mask)
    bin_image[rr[valid], cc[valid]] = 1  # draw the hair on the image

    return polygon_points, bbox


def draw_my_thick_bezier_curve_old(bin_image, s1_point, m_point, s2_direction, e_point, thickness, weight=3,
                                   neg_mask=None):
    points = [s1_point, m_point, e_point]
    curve = bezier_curve(points, 50)  # weight controls the resolution of the curve

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
    rr, cc = polygon(np.append(upper_curve[:, 1], lower_curve[::-1, 1]),
                     np.append(upper_curve[:, 0], lower_curve[::-1, 0]))

    if len(cc) == 0:
        return [], []

    empty_bin = np.zeros_like(bin_image)
    valid = (rr >= 0) & (rr < empty_bin.shape[0]) & (cc >= 0) & (cc < empty_bin.shape[1])
    empty_bin[rr[valid], cc[valid]] = 1  # draw the hair on empty for annotation
    polygon_points, bbox = get_polygons_bbox_from_bin_image(empty_bin, neg_mask)
    bin_image[rr[valid], cc[valid]] = 1  # draw the hair on the image

    return polygon_points, bbox


def draw_root_end_bezier_curve(bin_image, s1, s2, direction, length=20):
    cp1 = (s1 + direction * length).astype(int)
    cp2 = (s2 + direction * length).astype(int)
    end_point = ((s1 + s2) // 2 + direction * length).astype(int)

    points1 = [s1, cp1, end_point]
    points2 = [s2, cp2, end_point]

    curve1 = bezier_curve(points1)
    curve2 = bezier_curve(points2)

    # Draw the line between s1 and s2
    line_points = np.linspace(s1, s2, 100).astype(int)

    # Combine the curves and line points for the polygon
    polygon_points = np.vstack([curve1, curve2[::-1], [s2], [s1]])

    # Fill the polygon on the binary image
    rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0])
    valid = (rr >= 0) & (rr < bin_image.shape[0]) & (cc >= 0) & (cc < bin_image.shape[1])
    bin_image[rr[valid], cc[valid]] = 1  # Set pixels within polygon to 1


    return bin_image




def draw_single_root_end(bin_image, p1, p2, direction, length=10, thickness_=5, hair_crazy_=100):
    midpoint = (p1 + p2) // 2
    end_point = (midpoint + direction * length).astype(int)
    # cp = (midpoint + direction * (length - np.random.normal(length / 2, length / 8))).astype(int)
    draw_my_thick_bezier_curve_old(bin_image, midpoint, end_point, end_point, thickness_, hair_crazy_)


def get_upper_lower_curve(curve, thickness_profile):
    dx = np.gradient(curve[:, 0])
    dy = np.gradient(curve[:, 1])
    normals = np.column_stack([-dy, dx])
    norm_lengths = np.linalg.norm(normals, axis=1)

    # Normalize normals and apply adjusted thickness
    normals /= norm_lengths[:, np.newaxis]
    normals *= thickness_profile[:, np.newaxis]

    upper_curve = curve + normals
    lower_curve = curve - normals
    return lower_curve, upper_curve


def get_straight_thick_line(curve, thickness):
    if len(curve) == 1:
        point = curve[0]
        return np.array([point]), np.array([point])
    elif len(curve) == 2:
        start, end = curve
        direction = end - start
        direction = np.array([-direction[1], direction[0]])
        direction = direction / np.linalg.norm(direction) * thickness / 2
        upper_curve = np.array([start + direction, end + direction])
        lower_curve = np.array([start - direction, end - direction])
        return lower_curve, upper_curve
