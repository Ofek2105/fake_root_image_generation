import cv2
import numpy as np
import matplotlib.pyplot as plt

RIGHT_TAIL_THRESHOLD = 0.75


def thresholds_comparing(img, plot_=True, save_path=None):
    """
    Finds thresholds using various algorithms and displays them visually along with the thresholded images.

    Args:
        img: The input image (BGR format).
        plot_: Boolean flag to enable visualization (default: True).
        save_path: Path to save the figure (default: None).

    Returns:
        A dictionary containing threshold values from different algorithms.
    """
    results = {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = gray.flatten()

    counts, bins = np.histogram(vals, range(256))
    counts = counts / np.max(counts)

    # Otsu's threshold
    otsu_th, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    results["Otsu"] = int(otsu_th)

    # Peak-based right-tail threshold
    peak_x = np.argmax(counts)
    right_hist = counts[peak_x:]
    right_hist_sum = np.sum(right_hist)
    sum_ = 0
    for i in range(len(right_hist)):
        sum_ += right_hist[i]
        if sum_ / right_hist_sum > RIGHT_TAIL_THRESHOLD:
            break
    pos = peak_x + i
    results["Our"] = pos

    # Adaptive thresholding (mean C)
    AM_TH = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 6)
    results["Adaptive Mean"] = None

    # Adaptive thresholding (Gaussian C)
    AG_TH = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
    results["Adaptive Gaussian"] = None

    # Triangle method
    left_diff = np.diff(counts[:peak_x])  # Left side of the peak
    right_diff = np.diff(counts[peak_x:])  # Right side of the peak

    # Combine the differences from both sides
    if len(left_diff) > len(right_diff):
        left_diff = left_diff[:len(right_diff)]
    elif len(right_diff) > len(left_diff):
        right_diff = right_diff[:len(left_diff)]

    # Calculate the perpendicular distances for the triangle method
    valley_index = np.argmax(left_diff + right_diff) + peak_x
    results["Triangle"] = bins[valley_index]

    if plot_:
        # Prepare the thresholded images
        _, our_th_img = cv2.threshold(gray, pos, 255, cv2.THRESH_BINARY)
        _, triangle_th_img = cv2.threshold(gray, int(results["Triangle"]), 255, cv2.THRESH_BINARY)

        # Plot thresholded images in a 2x3 grid
        fig1, axs1 = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('Thresholding Comparisons', fontsize=20)

        # Plot original image
        axs1[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs1[0, 0].set_title('Original Image', fontsize=15)
        axs1[0, 0].axis('off')

        # Plot Otsu's threshold image
        axs1[0, 1].imshow(otsu_threshold, cmap='gray')
        axs1[0, 1].set_title(f'Otsu Threshold (T={results["Otsu"]})', fontsize=15)
        axs1[0, 1].axis('off')

        # Plot custom threshold image
        axs1[0, 2].imshow(our_th_img, cmap='gray')
        axs1[0, 2].set_title(f'Our Threshold (T={results["Our"]})', fontsize=15)
        axs1[0, 2].axis('off')

        # Plot triangle method threshold image
        axs1[1, 0].imshow(triangle_th_img, cmap='gray')
        axs1[1, 0].set_title(f'Triangle Threshold (T={int(results["Triangle"])})', fontsize=15)
        axs1[1, 0].axis('off')

        # Plot Adaptive Mean threshold image
        axs1[1, 1].imshow(AM_TH, cmap='gray')
        axs1[1, 1].set_title('Adaptive Mean Threshold', fontsize=15)
        axs1[1, 1].axis('off')

        # Plot Adaptive Gaussian threshold image
        axs1[1, 2].imshow(AG_TH, cmap='gray')
        axs1[1, 2].set_title('Adaptive Gaussian Threshold', fontsize=15)
        axs1[1, 2].axis('off')

        plt.tight_layout(pad=4.0, rect=[0.15, 0, 1, 0.95])  # Add padding between images and adjust title space
        if save_path:
            plt.savefig(save_path + "_thresholded_images.png", bbox_inches='tight')
        plt.show()

        # Plot the histogram and thresholds in a separate figure
        fig2, ax2 = plt.subplots(figsize=(18, 10))  # Increased the figure size for better obscure
        bar_list = ax2.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')

        # Highlight peak
        ax2.axvline(x=peak_x, color='k', linestyle='-', label=f'Peak x={peak_x}', alpha=0.5)

        # Add labels for each threshold
        for i, (name, threshold) in enumerate(results.items()):
            c_str = str(i + 1)
            if threshold is not None:
                ax2.text(threshold + 2, np.max(counts) - i * 0.05, f'{name}', fontsize=12,
                         bbox=dict(facecolor='C' + c_str, alpha=0.8))
                ax2.axvline(x=threshold, color='C' + c_str, linestyle='-', label=f'{name} x={threshold}', alpha=0.5)

        ax2.set_xlim([-0.5, 255.5])
        ax2.set_title('Histogram and Thresholds', fontsize=18)
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + "_histogram.png", bbox_inches='tight')
        plt.show()

    return results


def test_adaptive_thresholds(img, mean_params, gaussian_params, inverted=False, block_size=11, C_vals=None,
                             save_path=None):
    """
    Tests different parameters for Adaptive Mean and Adaptive Gaussian thresholding and displays them.

    Args:
        img: The input image (BGR format).
        mean_params: List of `C` parameters for Adaptive Mean thresholding.
        gaussian_params: List of `C` parameters for Adaptive Gaussian thresholding.
        block_size: The block size used for the adaptive thresholding (default: 11).
        C_vals: List of C values to subtract for each thresholding (default: None).
        save_path: Path to save the figure (default: None).

    Returns:
        None
    """
    cv2_th_enum = cv2.THRESH_BINARY if inverted else cv2.THRESH_BINARY_INV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_images = len(mean_params)

    if C_vals is None:
        C_vals = range(-10, 10, 2)  # Example C values if not provided

    # Set up the plot with a 3x10 grid (1 for original image, 10 for Adaptive Mean, 10 for Adaptive Gaussian)
    fig, axs = plt.subplots(3, n_images, figsize=(30, 12))
    fig.suptitle(f'Adaptive Thresholding Parameter Testing\n Block size {block_size}', fontsize=20)

    # Original image at the top center
    middle = n_images // 2
    axs[0, middle].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, middle].set_title('Original Image', fontsize=15)
    axs[0, middle].axis('off')

    # Adaptive Mean thresholding with different C values
    for i, C in enumerate(mean_params):
        AM_TH = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2_th_enum, block_size, C)
        axs[1, i].imshow(AM_TH, cmap='gray')
        axs[1, i].set_title(f'Ada Mean C={C}', fontsize=12)
        axs[1, i].axis('off')

    # Adaptive Gaussian thresholding with different C values
    for i, C in enumerate(gaussian_params):
        AG_TH = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2_th_enum, block_size, C)
        axs[2, i].imshow(AG_TH, cmap='gray')
        axs[2, i].set_title(f'Ada Gauss C={C}', fontsize=12)
        axs[2, i].axis('off')

    # Hide any unused subplots
    for ax in axs.flatten():
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to fit title and prevent overlap
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    image_paths = [r'../TEST DATA/base/arb_lr.png',
                   r'../TEST DATA/base/bell_lr.jpg']

    mean_params = list(range(2, 10, 2))
    gaussian_params = mean_params

    for img_path in image_paths:
        img = cv2.imread(img_path)
        # test_adaptive_thresholds(img, mean_params, gaussian_params, inverted=True, block_size=21)
        thresholds_comparing(img)
