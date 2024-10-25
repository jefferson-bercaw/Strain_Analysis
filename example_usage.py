## This script shows how to perform strain analysis automatically by one example

import numpy as np

from utils.strain_utils import average_point_cloud, remove_outer_boundaries
from utils.registration_utils import move_point_cloud, compute_asd

if __name__ == "__main__":

    start_path = 