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
  plt.text(otsu_threshold, plt.ylim()[1] * 0.6, 'OTSU Threshold', rotation=90, color='red', verticalalignment='center')

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
    file_name = fr"C:\Users\Ofek\Projects\OferHadar\RootHairsAnalysis\TEST DATA\binary\bell_sr_x8.png"
    cv2.imwrite(file_name, black_image)
    plt.imshow(black_image, cmap='gray')
    plt.show()
  return black_image


def get_hairs_contours(binary_image, filename=None, plot_=False, truth_count="", suptitle_=None):
  hairs_bin_image = get_hairs_tips_bin_image(binary_image)
  hairs_dilated = cv2.dilate(hairs_bin_image.copy(), np.ones((3, 3), np.uint8), iterations=1)

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
  kernel = np.ones((3, 3), np.uint8)
  bin_image_morph = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
  bin_image_morph = cv2.morphologyEx(bin_image_morph, cv2.MORPH_CLOSE, kernel)
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


def find_right_tail_threshold(img, plot_=False):
  vals = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
  counts, bins = np.histogram(vals, range(256))

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
    y_text = np.max(counts) // 4
    bar_list = plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    for j in range(pos, 255):
      bar_list[j].set_color('red')
    plt.xlim([-0.5, 255.5])
    plt.text(peak_x + 2, y_text, f'Peak\n', fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
    plt.text(peak_x + i + 2, y_text + 300, f'{int((1 - RIGHT_TAIL_THRESHOLD) * 100)}% sum\nfrom peak', fontsize=10,
             bbox=dict(facecolor='red', alpha=0.8))

    plt.axvline(x=pos, color='k', linestyle='-', label=f'x={pos}')
    plt.axvline(x=peak_x, color='k', linestyle='-', label=f'x={pos}')
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


def get_root_and_hairs_mask_OG(image, plot_report_):
  x_pos = find_right_tail_threshold(image, plot_=plot_report_)
  print(x_pos)
  chosen_range = (int(x_pos), 255)
  bin_image = img_intensity_range(image, chosen_range, plot=plot_report_)
  root_image = get_largest_contur(bin_image, plot_report_)
  return root_image


def get_root_and_hairs_mask_OTSU(image, plot_report, blur_=None):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  if blur_ is not None:
    gray = cv2.GaussianBlur(gray, (blur_[0], blur_[1]), blur_[2])

  value, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  print(value)
  root_image = get_largest_contur(binary_image, plot_report)
  return root_image


def get_title(full_path):
  title_name = full_path.split('\\')[-1].split('.')[0]
  return " ".join(title_name.split('_'))


def save_plot(image, title, path):
  plt.imshow(image, cmap='gray')
  plt.title(title, fontsize=25)
  plt.axis('off')
  plt.savefig(path)


def get_image_hair_count(image_path, plot_report=False, blur=None):
  # save_path = r'TEST DATA/results/lr/lr_bell.png'
  # file_title = get_title(image_path)
  sup_tit = ""
  if blur is not None:
    sup_tit = f'OSTU algo blur kernal ({blur[0]}, {blur[1]}) sigma {blur[2]}'
  image = cv2.imread(image_path)
  root_bin_image = get_root_and_hairs_mask_OG(image, True)
  hair_contours = get_hairs_contours(root_bin_image, filename=None, plot_=plot_report, suptitle_=sup_tit)
  hair_num = len(hair_contours)
  # original_image_with_contours = draw_contours(image, hair_contours, plot_report)
  # bin_image_with_contours = draw_contours(root_bin_image, hair_contours, (0, 255, 0), plot_report)

  # save_plot(bin_image_with_contours, f'Bell Pepper Num = {hair_num}', save_path)
  return hair_num


if __name__ == '__main__':
  image_paths = [r'../TEST DATA/base_path/arb_lr.png',
                 r'../TEST DATA/base_path/bell_lr.jpg']

  image_paths = [r'../TEST DATA/BELL PEPEER/SR_P1_X8.png']

  # plot_histogram_with_thresholds(image_paths[0], our_threshold=147, otsu_threshold=127)
  # plot_histogram_with_thresholds(image_paths[1], our_threshold=102, otsu_threshold=88)


  # blur_props = [None, (3, 3, 0), (5, 5, 0), (11, 11, 0)]
  # blur_props = [None]
  # for image_path in image_paths:
#   for blur_prop in blur_props:
  n_hairs = get_image_hair_count(image_paths[0], plot_report=True)
  # n_hairs = get_image_hair_count(image_paths[1], plot_report=True)


  # base_path = r'TEST DATA\\ARBIDIOPSIS'
  # for file_name in os.listdir(base_path):
  #   full_path = os.path.join(base_path, file_name)
  #   n_hairs = get_image_hair_count(full_path, plot_report=False)
  #   print(f'{file_name} Hairs number: {n_hairs}')
