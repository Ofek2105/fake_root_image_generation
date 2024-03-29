import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

channel_name = {0: 'blue', 1: 'green', 2: 'red'}
channel_code = {0: 'b', 1: 'g', 2: 'r'}


def img_intensity_range(img, bins, plot=False):
  img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # img_ch = img[:, :, chan_num]
  mask = np.zeros_like(img_ch)
  mask[(img_ch >= bins[0]) & (img_ch <= bins[1])] = 255
  # mask[(img_ch < bins[0]) & (img_ch > bins[1])] = 0
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
    plt.imshow(black_image, cmap='gray')
    plt.show()
  return black_image


def erode_sub_count(binary_image, filename=None, plot_=False):
  hairs_bin_image = get_hairs_tips_bin_image(binary_image)
  hairs_dilated = cv2.dilate(hairs_bin_image.copy(), np.ones((3, 3), np.uint8), iterations=3)

  image_hairs_highlight = draw_overlay_on_canvas(binary_image, hairs_dilated)
  contours, _ = cv2.findContours(hairs_bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  fig, ax = plt.subplots(1, 3, figsize=(45, 15))
  font_size = 50

  if plot_:
    ax[0].imshow(binary_image, cmap='gray')
    ax[0].set_title('Base Binary image', fontsize=font_size)

    ax[1].imshow(hairs_dilated, cmap='gray')
    ax[1].set_title('Found Hair Tips', fontsize=font_size)

    ax[2].imshow(image_hairs_highlight, cmap='gray')
    ax[2].set_title(f'SR (our method) x8 num={len(contours)}', fontsize=font_size)
    plt.axis('off')
    plt.show()

  if filename is not None:
    file1 = rf'res\mine_good\{filename}_bin.png'
    file2 = rf'res\mine_good\{filename}_bin_with_tips.png'
    cv2.imwrite(file1, binary_image)
    cv2.imwrite(file2, image_hairs_highlight)

  return len(contours), hairs_bin_image, contours


def get_hairs_tips_bin_image(binary_image):
  kernel = np.ones((3, 3), np.uint8)
  erosion = cv2.erode(binary_image, kernel, iterations=1)
  dilation = cv2.dilate(erosion, kernel, iterations=1)
  dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
  hairs = cv2.subtract(binary_image, dilation)
  return hairs


def draw_overlay_on_canvas(canvas_image, overlay_image):
  canvas_image[canvas_image > 0] = 255
  canvas_color = cv2.cvtColor(canvas_image, cv2.COLOR_GRAY2BGR)
  y, x = np.where(overlay_image > 0)
  canvas_color[y, x] = [0, 255, 0]

  return canvas_color


def find_right_tail_threshold(img, percentage, plot_=False):
  vals = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
  counts, bins = np.histogram(vals, range(256))

  peak_x = np.argmax(counts)
  right_hist = counts[peak_x:]
  right_hist_sum = np.sum(right_hist)
  sum_ = 0
  for i in range(len(right_hist)):
    sum_ += right_hist[i]
    if sum_ / right_hist_sum > percentage:
      break

  pos = peak_x + i

  if plot_:
    y_text = np.max(counts) // 4
    bar_list = plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    for j in range(pos, 255):
      bar_list[j].set_color('red')
    plt.xlim([-0.5, 255.5])
    plt.text(peak_x + 2, y_text, f'Peak\n100%', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
    plt.text(peak_x + i + 2, y_text, f'{int((1 - percentage) * 100)}%', fontsize=10,
             bbox=dict(facecolor='red', alpha=0.8))

    plt.axvline(x=pos, color='k', linestyle='-', label=f'x={pos}')
    plt.axvline(x=peak_x, color='k', linestyle='-', label=f'x={pos}')
    plt.show()
  return pos


def draw_contours(original_image, contours_, plot_=False):
  for contour in contours_:
    color = np.random.randint(0, 255, size=3).tolist()  # Generate a random color
    cv2.drawContours(original_image, [contour], -1, color, 2)  # -1 means drawing all contours

  if plot_:
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

  return original_image


def get_image_hair_count(image_path, plot_report=False):
  image = cv2.imread(image_path)
  right_tail_percent_th = 0.75

  # hist_peak, std = find_histogram_peak_and_std(image, chosen_channel)
  x_pos = find_right_tail_threshold(image, right_tail_percent_th, plot_=plot_report)
  chosen_range = (int(x_pos), 255)

  bin_image = img_intensity_range(image,
                                  chosen_range, plot=plot_report)
  root_image = get_largest_contur(bin_image, plot_report)
  hairs_num, _, hair_contours = erode_sub_count(root_image, filename=None, plot_=plot_report)

  original_image_with_contours = draw_contours(image, hair_contours, plot_report)

  return hairs_num


if __name__ == '__main__':
  n_hairs = get_image_hair_count(r'TEST DATA/BELL PEPEER/SR_P1_X8.png', plot_report=True)
  print(f'Hairs number: {n_hairs}')
