import os
import time

import matplotlib.pyplot as plt
import cv2
import numpy as np

RIGHT_TAIL_THRESHOLD = 0.75
id_ = 0


def plot_histogram_with_thresholds(image_path, our_threshold, otsu_threshold):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to read.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.max()  # Normalize histogram

    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='gray')
    plt.fill_between(range(256), hist, color='gray', alpha=0.5)

    plt.axvline(x=our_threshold, color='blue', label=f'Our Threshold: {our_threshold}')
    plt.text(our_threshold, plt.ylim()[1] * 0.8, 'Our Threshold', rotation=90, color='blue', verticalalignment='center')
    plt.axvline(x=otsu_threshold, color='red', label=f'OTSU Threshold: {otsu_threshold}')
    plt.text(otsu_threshold, plt.ylim()[1] * 0.6, 'OTSU Threshold', rotation=90, color='red',
             verticalalignment='center')

    # Enhancing the plot
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Counts')
    plt.title('Histogram with Thresholds')
    plt.legend()
    plt.grid(True)
    plt.show()


def img_intensity_range(img, bins, plot=False):
    img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_ch)
    mask[(img_ch >= bins[0]) & (img_ch <= bins[1])] = 255

    if plot:
        plt.title(f'only {bins} range pixels')
        plt.imshow(mask, cmap='gray')
        plt.show()
    return mask


def get_largest_contur(bin_image_, plot_=False):
    contours, _ = cv2.findContours(bin_image_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_image = np.zeros(bin_image_.shape, dtype=np.uint8)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(black_image, [largest_contour], -1, 255, -1)

    if plot_:
        file_name = fr"C:\Users\Ofek\Projects\OferHadar\RootHairsAnalysis\results\operated\binary_.png"
        cv2.imwrite(file_name, black_image)
        plt.imshow(black_image, cmap='gray')
        plt.show()
    return black_image


def get_hairs_contours(binary_image, filename=None, plot_=False, truth_count="", suptitle_=None):
    hairs_bin_image = get_hairs_tips_bin_image(binary_image)
    hairs_dilated = cv2.dilate(hairs_bin_image.copy(), np.ones((3, 3), np.uint8), iterations=2)

    hairs_contours, _ = cv2.findContours(hairs_bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if plot_ or filename:

        fig, ax = plt.subplots(1, 2, figsize=(45, 15))
        font_size = 50
        if suptitle_ is not None:
            fig.suptitle(f'{suptitle_}', fontsize=font_size)

        ax[0].imshow(binary_image, cmap='gray')
        if truth_count != "":
            truth_count = f'hair count={truth_count}'
        ax[0].set_title(f'Base Binary image {truth_count}', fontsize=font_size)

        # ax[1].imshow(hairs_dilated, cmap='gray')
        # ax[1].set_title('Found Hair Tips', fontsize=font_size)
        image_hairs_highlight = draw_overlay_on_canvas(binary_image, hairs_dilated)
        ax[1].imshow(image_hairs_highlight)
        ax[1].set_title(f'predicted hair count = {len(hairs_contours)}', fontsize=font_size)
        plt.axis('off')

    if filename is not None:
        plt.savefig(filename)

    if plot_:
        plt.show()

    return hairs_contours


def get_hairs_tips_bin_image(binary_image):
    # temp_path = r"C:\Users\Ofek\Projects\OferHadar\RootHairSementationModel\save_dump\haircounting"
    kernel = np.ones((3, 3), np.uint8)

    # cv2.imwrite(f"{temp_path}\\im_1_base.jpg", binary_image)

    bin_image_morph = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # cv2.imwrite(f"{temp_path}\\im_2_open.jpg", bin_image_morph)

    bin_image_morph = cv2.morphologyEx(bin_image_morph, cv2.MORPH_CLOSE, kernel)

    # cv2.imwrite(f"{temp_path}\\im_3_close.jpg", bin_image_morph)

    hairs = cv2.subtract(binary_image, bin_image_morph)
    # hairs = cv2.erode(hairs, kernel, iterations=1)
    hairs = cv2.dilate(hairs, kernel, iterations=2)
    return hairs


def draw_overlay_on_canvas(canvas_image, overlay_image):
    canvas_image[canvas_image > 0] = 255
    canvas_color = cv2.cvtColor(canvas_image, cv2.COLOR_GRAY2BGR)
    y, x = np.where(overlay_image > 0)
    canvas_color[y, x] = [0, 255, 0]

    return canvas_color


def find_right_tail_threshold(img, TH_value=None, plot_=False):
    vals = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
    counts, bins = np.histogram(vals, range(256))
    counts = counts / np.max(counts)
    peak_x = np.argmax(counts)
    right_hist = counts[peak_x:]
    right_hist_sum = np.sum(right_hist)
    sum_ = 0
    for i in range(len(right_hist)):
        sum_ += right_hist[i]
        if sum_ / right_hist_sum > RIGHT_TAIL_THRESHOLD:
            break

    pos = peak_x + i

    if plot_:
        y_text = np.max(counts) / 2
        bar_list = plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        for j in range(pos, 255):
            bar_list[j].set_color('red')
        plt.xlim([-0.5, 255.5])
        plt.text(peak_x + 2, y_text + 0.15, f'Peak\n', color='white', fontsize=10, bbox=dict(facecolor='k', alpha=0.8))

        plt.axvline(x=peak_x, color='k', linestyle='-', label=f'Peak x={peak_x}', alpha=0.5)

        plt.text(peak_x + i + 2, y_text - 0.3, f'{int((1 - RIGHT_TAIL_THRESHOLD) * 100)}% sum\nfrom peak', fontsize=10,
                 bbox=dict(facecolor='r', alpha=0.8))
        plt.axvline(x=pos, color='r', linestyle='-', label=f'Our x={pos}')

        if TH_value is not None:
            plt.text(TH_value + 2, y_text - 0.1, f'OTSU', fontsize=10, bbox=dict(facecolor='blue', alpha=0.8))
            plt.axvline(x=TH_value, color='b', linestyle='-', label=f'OTSU x={TH_value}')

        plt.legend()
        plt.show()
    return pos


def draw_contours(original_image, contours_, color=None, plot_=False):
    for contour in contours_:
        if color is None:
            color = np.random.randint(0, 255, size=3).tolist()  # Generate a random color
        cv2.drawContours(original_image, [contour], -1, color, 2)  # -1 means drawing all contours

    if plot_:
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return original_image


def get_root_and_hairs_mask_OG(image, plot_report_, return_TH_value=False):
    x_pos = find_right_tail_threshold(image, plot_=plot_report_)
    print(f'original TH value {x_pos}')
    chosen_range = (int(x_pos), 255)
    bin_image = img_intensity_range(image, chosen_range, plot=plot_report_)
    root_image = get_largest_contur(bin_image, plot_report_)
    if return_TH_value:
        return root_image, x_pos
    return root_image


def get_root_and_hairs_mask_OTSU(image, plot_report, blur_=None, return_TH_value=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_ is not None:
        gray = cv2.GaussianBlur(gray, (blur_[0], blur_[1]), blur_[2])

    value, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    root_image = get_largest_contur(binary_image, plot_report)
    print(f'original OTSU value {value}')

    if plot_report:
        find_right_tail_threshold(image, TH_value=int(value), plot_=plot_report)

    if return_TH_value:
        return root_image, value

    return root_image


def find_triangle_threshold(image, plot_report_=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vals = gray.flatten()
    counts, bins = np.histogram(vals, range(256))
    counts = counts / np.max(counts)

    peak_x = np.argmax(counts)
    left_diff = np.diff(counts[:peak_x])
    right_diff = np.diff(counts[peak_x:])

    if len(left_diff) > len(right_diff):
        left_diff = left_diff[:len(right_diff)]
    elif len(right_diff) > len(left_diff):
        right_diff = right_diff[:len(left_diff)]

    valley_index = np.argmax(left_diff + right_diff) + peak_x
    triangle_threshold = bins[valley_index]

    if plot_report_:
        plt.figure(figsize=(10, 5))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.axvline(x=triangle_threshold, color='r', linestyle='-', label=f'Triangle TH x={triangle_threshold}')
        plt.xlim([-0.5, 255.5])
        plt.legend()
        plt.show()

    return triangle_threshold


def get_root_and_hairs_mask_Triangle(image, plot_report_=False, return_TH_value=False):
    triangle_th = find_triangle_threshold(image, plot_report_)
    print(f'Triangle TH value {triangle_th}')
    chosen_range = (int(triangle_th), 255)
    bin_image = img_intensity_range(image, chosen_range, plot=plot_report_)
    root_image = get_largest_contur(bin_image, plot_report_)
    if return_TH_value:
        return root_image, triangle_th
    return root_image


def get_root_and_hairs_mask_AdaptiveMean(image, plot_report_=False, return_TH_value=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = 11  # Default block size
    C = 2  # Default C value
    adaptive_mean_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, block_size, C)
    root_image = get_largest_contur(adaptive_mean_th, plot_report_)
    if plot_report_:
        plt.figure(figsize=(10, 5))
        plt.title(f'Adaptive Mean TH (C={C})')
        plt.imshow(adaptive_mean_th, cmap='gray')
        plt.axis('off')
        plt.show()
    if return_TH_value:
        return root_image, C
    return root_image


def get_root_and_hairs_mask_AdaptiveGaussian(image, plot_report_=False, return_TH_value=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = 11  # Default block size
    C = 2  # Default C value
    adaptive_gaussian_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, block_size, C)
    root_image = get_largest_contur(adaptive_gaussian_th, plot_report_)
    if plot_report_:
        plt.figure(figsize=(10, 5))
        plt.title(f'Adaptive Gaussian TH (C={C})')
        plt.imshow(adaptive_gaussian_th, cmap='gray')
        plt.axis('off')
        plt.show()
    if return_TH_value:
        return root_image, C
    return root_image


def get_title(full_path):
    title_name = full_path.split('\\')[-1].split('.')[0]
    return " ".join(title_name.split('_'))


def save_plot(image, title, path):
    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=25)
    plt.axis('off')
    plt.savefig(path)


def get_image_hair_count(image_path, th_algo="Our", plot_report=False, blur=None, save_bin=None):
    # save_path = r'TEST DATA/results/lr/lr_bell.png'
    # file_title = get_title(image_path)
    sup_tit = ""
    if blur is not None:
        sup_tit = f'OSTU algo blur kernal ({blur[0]}, {blur[1]}) sigma {blur[2]}'
    image = cv2.imread(image_path)

    if th_algo == "Our":
        root_bin_image = get_root_and_hairs_mask_OG(image, plot_report)
    elif th_algo == "OTSU":
        root_bin_image = get_root_and_hairs_mask_OTSU(image, plot_report, blur_=blur)
    elif th_algo == "Triangle":
        root_bin_image = get_root_and_hairs_mask_Triangle(image, plot_report)
    elif th_algo == "Adaptive Mean":
        root_bin_image = get_root_and_hairs_mask_AdaptiveMean(image, plot_report)
    elif th_algo == "Adaptive Gaussian":
        root_bin_image = get_root_and_hairs_mask_AdaptiveGaussian(image, plot_report)
    else:
        raise ValueError(f"Unknown thresholding algorithm: {th_algo}")

    hair_contours = get_hairs_contours(root_bin_image, filename=save_bin, plot_=plot_report, suptitle_=sup_tit)
    hair_num = len(hair_contours)
    # if save_bin is not None:
    #     cv2.imwrite(save_bin, root_bin_image)
    # original_image_with_contours = draw_contours(image, hair_contours, plot_report)
    # bin_image_with_contours = draw_contours(root_bin_image, hair_contours, (0, 255, 0), plot_report)

    # save_plot(bin_image_with_contours, f'Bell Pepper Num = {hair_num}', save_path)
    return hair_num


def set_right_tail_threshold(value):
    global RIGHT_TAIL_THRESHOLD
    RIGHT_TAIL_THRESHOLD = value


def process_and_plot_influence(image_folder, num_images=10, thresholds=[0.65, 0.7, 0.75, 0.8, 0.9], plot_report=False):
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)[:num_images]]

    for idx, image_path in enumerate(images):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 5))
        fig.suptitle(f"Effect of RIGHT_TAIL_THRESHOLD on Image {idx + 1}", fontsize=16)

        for i, threshold in enumerate(thresholds):
            set_right_tail_threshold(threshold)
            bin_image = get_root_and_hairs_mask_OG(image, plot_report)  # Simulate using the threshold in the algorithm
            axes[i].imshow(bin_image, cmap='gray')
            axes[i].set_title(f"Threshold: {round(1 - threshold, 2)}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

if __name__ == '__main__':
    image_paths = [r'../TEST DATA/base/arb_lr.png',
                   r'../TEST DATA/base/bell_lr.jpg']

    # image_paths = [r'../TEST DATA/ARBIDIOPSIS/arb_sr_x2.png']
    image_paths = [r'../results/type_2/zoomIn.png']
    image_paths = [r'../results/type_1/SR_P1_X4.png']

    # image_paths = [r'../TEST DATA/BELL PEPEER/SR_P1_X2.png']
    n_hairs = get_image_hair_count(image_paths[0], th_algo="Our", plot_report=True)
    # plot_histogram_with_thresholds(image_paths[0], our_threshold=147, otsu_threshold=127)
    # plot_histogram_with_thresholds(image_paths[1], our_threshold=102, otsu_threshold=88)

    # =====
    # folder_path = r'C:\Users\Ofek\Desktop\arb_root_images'
    # total_time = 0
    # file_count = 0
    # process_and_plot_influence(folder_path, num_images=10)
    # ======

    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     start_time = time.time()
    #     # n_hairs = get_image_hair_count(file_path, plot_report=False, save_bin=f"../results/bin_arb_roots/{file_name}")
    #     n_hairs = get_image_hair_count(file_path, plot_report=False, save_bin=None)
    #     elapsed_time = time.time() - start_time
    #     total_time += elapsed_time
    #     file_count += 1
    #     # print(f"{file_name}: {n_hairs} hairs counted in {elapsed_time:.2f} seconds.")
    #
    # average_time = total_time / file_count if file_count > 0 else 0
    # print(f"\nProcessed {file_count} files.")
    # print(f"Average time to calculate root hair count: {average_time} seconds.")
    # # n_hairs = get_image_hair_count(image_paths[1], plot_report=True)

