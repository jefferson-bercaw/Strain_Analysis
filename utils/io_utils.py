## This file contains helper functions for input and outputs

import open3d as o3d
import numpy as np


def load_pcd_file(filename):
    """This function loads a point cloud from a .pcd file and returns it as a numpy array.
    Inputs:
        filename: The path to the .pcd file
    Returns:
        points: The point cloud as a numpy array (nx3)
    """
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points) * 1000.0 # Convert to mm
    return points


def save_pcd_file(points, filename):
    """This function saves a point cloud to a .pcd file.
    Inputs:
        points: The point cloud as a numpy array (nx3)
        filename: The path to save the .pcd file
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    return
