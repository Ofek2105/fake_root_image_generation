import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def adaptive_thresholding(img, block_size=3, sub_const=3, mode_='mean'):
  if mode_ == 'mean':
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, sub_const)
  elif mode_ == 'gaussian':
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, sub_const)

def plot_thresholding_parameters():
  image = cv2.imread(r'../results/type_3/arb_sr_x4.png')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

  plt.figure(figsize=(20, 15))

  block_sizes = [11, 21, 31, 41]
  sub_consts = [2, 5, 10, 15]
  modes = [('Mean', cv2.ADAPTIVE_THRESH_MEAN_C), ('Gaussian', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)]
  output_dir = "savedump"

  plot_idx = 1
  for mode_name, mode in modes:
    for block_size in block_sizes:
      for sub_const in sub_consts:
        thresholded = cv2.adaptiveThreshold(image, 255, mode, cv2.THRESH_BINARY, block_size, sub_const)
        plt.subplot(len(modes), len(block_sizes) * len(sub_consts), plot_idx)
        plt.imshow(thresholded, cmap='gray')
        plt.title(f"{mode_name}\nBlock: {block_size}, C: {sub_const}")
        plt.axis('off')
        plot_idx += 1
        filename = f"{mode}_block{block_size}_c{sub_const}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, thresholded)

  plt.tight_layout()
  plt.show()


def main():
  image = cv2.imread(r'../results/type_3/arb_sr_x4.png')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
  block_size = [11, 13, 15]
  sub_th_consts = [7, 9, 11]
  mode = 'mean'
  fig, axes = plt.subplots(len(block_size), len(sub_th_consts), figsize=(40, 32))
  for row_i, block in enumerate(block_size):
    for col_i, sub_const in enumerate(sub_th_consts):
      bin_image = adaptive_thresholding(image, block_size=block, sub_const=sub_const,
                                        mode_=mode)

      ax = axes[row_i, col_i]
      ax.set_title(f'block size:{block}, sub const{sub_const}')
      ax.title.set_size(12)
      ax.imshow(bin_image, cmap='gray')  # Use grayscale color map for binary images
      ax.axis('off')  # Turn off axis
      # col += 1  # Move to the next column
  plt.subplots_adjust(hspace=0.1, wspace=0.1)
  plt.show()


if __name__ == "__main__":
  # main()
  plot_thresholding_parameters()

