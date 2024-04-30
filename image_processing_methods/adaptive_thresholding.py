import matplotlib.pyplot as plt
import cv2
import numpy as np


def adaptive_thresholding(img, block_size=3, sub_const=3, mode_='mean'):
  if mode_ == 'mean':
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, sub_const)
  elif mode_ == 'gaussian':
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, sub_const)


if __name__ == "__main__":

  image = cv2.imread(r'../res/type_3/arb_sr_x4.png')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
  block_size = [11, 13, 15]
  sub_th_consts = [7, 9, 11]
  mode = 'gaussian'

  fig, axes = plt.subplots(len(block_size), len(sub_th_consts), figsize=(40, 32))

  for row_i, block in enumerate(block_size):
    for col_i, sub_const in enumerate(sub_th_consts):
      bin_image = adaptive_thresholding(image, block_size=block, sub_const=sub_const,
                                        mode_=mode)

      ax = axes[row_i, col_i]
      ax.set_title(f'block size:{block}, sub const{sub_const}')
      ax.title.set_size(48)
      ax.imshow(bin_image, cmap='gray')  # Use grayscale color map for binary images
      ax.axis('off')  # Turn off axis
      # col += 1  # Move to the next column
  plt.subplots_adjust(hspace=0.1, wspace=0.1)
  plt.show()
