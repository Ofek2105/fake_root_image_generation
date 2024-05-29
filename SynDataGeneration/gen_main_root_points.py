import numpy as np
import matplotlib.pyplot as plt


def generate_annular_init_points(N, rect_out_, rect_in_):
  out_s_col, out_s_row, out_end_col, out_end_row = rect_out_
  in_s_col, in_s_row, in_end_col, in_end_row = rect_in_
  all_points = []
  # Determine initial spacing based on the area of the annular region
  total_area = (out_end_col - out_s_col) * (out_end_row - out_s_row)
  inner_area = (in_end_col - in_s_col) * (in_end_row - in_s_row)
  annular_area = total_area - inner_area
  estimated_density = N / annular_area  # Points per unit area
  spacing = np.sqrt(1 / estimated_density)  # Approximate spacing based on density

  # Generate grid points considering spacing
  while len(all_points) < N:
    all_points.clear()  # Clear previous points for new grid resolution
    x_values = np.arange(out_s_col + spacing, out_end_col - spacing, spacing)
    y_values = np.arange(out_s_row + spacing, out_end_row - spacing, spacing)

    for x in x_values:
      for y in y_values:
        if not (in_s_col + spacing <= x <= in_end_col - spacing and in_s_row + spacing <= y <=
                in_end_row - spacing):
          all_points.append((x, y))

    # Increase the number of points by decreasing spacing if not enough points
    if len(all_points) < N:
      spacing *= 0.95  # Gradually decrease spacing to fit more points

  np.random.shuffle(all_points)  # Shuffle to randomize point selection
  points = np.array(all_points[:N])
  return points


def calculate_direction(start_point, rect_in_):
  in_s_col, in_s_row, in_end_col, in_end_row = rect_in_
  # Calculate the midpoint of the rectangle
  mid_x = (in_s_col + in_end_col) / 2
  mid_y = (in_s_row + in_end_row) / 2

  # Calculate the direction from start_point to the midpoint of the rectangle
  direction = np.degrees(np.arctan2(mid_y - start_point[1], mid_x - start_point[0]))

  return direction


def random_walk_line(start_point, rect_out_, rect_in_,
                     stop_chance=0.001, step_size=1,
                     momentum=0.9, steps=1000):

  x, y = start_point
  initial_direction = calculate_direction(start_point, rect_in_)
  direction = np.radians(initial_direction + np.random.uniform(-5, 5))  # Adding a random bias
  x_coords, y_coords = [x], [y]

  last_sign = 0
  consecutive_count = 0
  bias_adjustment_factor = 5  # higher means less looping

  for _ in range(steps):
    if np.random.rand() < stop_chance:
      break  # Random chance to stop the line

    mean_shift = -last_sign * consecutive_count * bias_adjustment_factor
    change = np.random.normal(0, scale=(1 - momentum))

    direction += change
    x_step = step_size * np.cos(direction)
    y_step = step_size * np.sin(direction)

    x += x_step
    y += y_step

    if not is_in_rect(x, y, rect_out_):
      break

    x_coords.append(x)
    y_coords.append(y)

  return x_coords, y_coords


def is_in_rect(x_coords, y_coords, rect_):
  s_col, s_row, end_col, end_row = rect_
  if s_col <= x_coords <= end_col and s_row <= y_coords <= end_row:
    return True
  return False


def generator_main_roots(N, minimum_length=50):
  """
  a generator that yields points. each is resembling a main root
  :param N: number of roots to generate
  :param rect_out_start: outer bound 1
  :param rect_out_end: outer bound 2
  :param rect_in_start: inner bound 1
  :param rect_in_end: inner bound 2
  :return: yields points
  """

  rect_out_ = (50, 50, 250, 250)
  delt = 20
  rect_in_ = (rect_out_[0] + delt, rect_out_[1] + delt, rect_out_[2] - delt, rect_out_[3] - delt)

  init_points = generate_annular_init_points(N, rect_out_, rect_in_)
  for init_point in init_points:
    x_coords, y_coords = random_walk_line(init_point, rect_out_, rect_in_)
    root_points = np.vstack((x_coords, y_coords)).T
    if len(root_points) < minimum_length:
      # print("redo main root")
      continue
    yield root_points


if __name__ == "__main__":
  rect_out = (20, 20, 80, 80)
  rect_in = (40, 40, 60, 60)

  for main_root_points in generator_main_roots(5):
    print(main_root_points.shape)
    plt.scatter(main_root_points[:, 0], main_root_points[:, 1])
    plt.show()
    break
