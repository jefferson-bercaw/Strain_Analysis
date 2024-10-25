## This script shows how to perform strain analysis automatically by one example

import numpy as np
import os

from utils.strain_utils import average_point_cloud, remove_outer_boundaries, calculate_thickness, calculate_strain
from utils.registration_utils import register_bones, move_points, compute_asd
from utils.io_utils import load_pcd_file, save_pcd_file

if __name__ == "__main__":

    ######## Inputs ########
    start_path = "R:\\DefratePrivate\\Obesity R01\\PointFilesForJefferson\\Tibia\\pcdNoNormals"

    # Perform on one plateau to start
    plateau = "Medial"
    pre_tib = load_pcd_file(os.path.join(start_path, "PreTibia.pcd"))
    post_tib = load_pcd_file(os.path.join(start_path, "PostTibia.pcd"))
    pre_cart = load_pcd_file(os.path.join(start_path, f"PreCart{plateau}.pcd"))
    post_cart = load_pcd_file(os.path.join(start_path, f"PostCart{plateau}.pcd"))

    ######## Registration ########
    # Register the post to the pre, if needed
    post_tib, transform = register_bones(pre_tib, post_tib, output_option=True)

    # Compute ASD between the bones, as a sanity check
    asd = compute_asd(pre_tib, post_tib)
    print(f"ASD between registered bones: {asd} mm")

    # Move the post cartilage to the pre cartilage, if needed
    post_cart = move_points(post_cart, transform)

    ######## Thickness Analysis ########
    # Compute thickness for both pre and post cartilage
    pre_thick = calculate_thickness(pre_tib, pre_cart)
    post_thick = calculate_thickness(post_tib, post_cart)

    # Calculate strain
    strain_points, strain = calculate_strain(pre_cart, pre_thick, post_cart, post_thick)

    # Remove outer boundaries
    strain_points, strain = remove_outer_boundaries(strain_points, strain, radius=5, n=7, percent=10)

    # Apply a moving average to the strain
    strain = average_point_cloud(strain_points, strain, radius=2.5)

    print("Mean Strain: ", np.mean(strain))

