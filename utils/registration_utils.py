## This file contains functionality for registration of two solid bones (i.e. tibiae)

import pickle
import os
import numpy as np
import open3d as o3d
import copy


def register_bones(fixed_points, moving_points, output_option):
    """This function registers the moving_points to the fixed_points. They are both nx3 ndarrays containing points in
    3D space for the two bones that are being registered. The output_option is a bool that determines whether or not
    you'd like to see the output point clouds displayed.

    Returns:
          pcd_moved: The moving point cloud after registration (nx3 ndarray)
          icp.transformation: The transformation matrix that was applied to the moving point cloud (4x4 ndarray)
    """

    # Create point clouds
    pcd_fixed = o3d.geometry.PointCloud()
    pcd_fixed.points = o3d.utility.Vector3dVector(fixed_points)

    pcd_moving = o3d.geometry.PointCloud()
    pcd_moving.points = o3d.utility.Vector3dVector(moving_points)

    ######## Perform RANSAC registration as a starting point ########
    # Downsample both point clouds
    voxel_size_d = 1.5  # Downsample so every point is this far apart from all other points
    p_fixed_d = pcd_fixed.voxel_down_sample(voxel_size=voxel_size_d)
    p_moving_d = pcd_moving.voxel_down_sample(voxel_size=voxel_size_d)

    # Compute normals
    search_radius_norm = voxel_size_d * 2
    p_fixed_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius_norm, max_nn=30))
    p_moving_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius_norm, max_nn=30))

    # Compute fpfh features for each point
    search_radius_feature = voxel_size_d * 5
    p_fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(p_fixed_d, o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_feature, max_nn=100))
    p_moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(p_moving_d, o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_feature, max_nn=100))

    # Execute
    distance_threshold = voxel_size_d * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(p_moving_d, p_fixed_d,
                                                                                      p_moving_fpfh, p_fixed_fpfh, True,
                                                                                      distance_threshold,
                                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                          False),
                                                                                      3,
                                                                                      [
                                                                                          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                                                                              distance_threshold)],

                                                                                      o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                          100000, 0.9999))

    # Move the moving point cloud
    p_moved = copy.deepcopy(pcd_moving)
    p_moved.transform(result.transformation)

    ######## Perform ICP Registration ########
    # Figure out rough estimate of voxel size
    distances = pcd_fixed.compute_nearest_neighbor_distance()

    # Parameters
    voxel_size = 0.2
    threshold = voxel_size * 8
    initial_moving_to_fixed = result.transformation  # Use RANSAC as a starting point

    # Execute
    icp = o3d.pipelines.registration.registration_icp(pcd_moving, pcd_fixed, threshold, initial_moving_to_fixed,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(
                                                          relative_fitness=1e-10,
                                                          relative_rmse=1e-10,
                                                          max_iteration=50000)
                                                      )

    # Move the moving point cloud
    pcd_moved = pcd_moving.transform(icp.transformation)

    if output_option:
        # Visualizing ICP transform
        pcd_moved.paint_uniform_color([1, 0, 0])
        pcd_fixed.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([pcd_moved, pcd_fixed], window_name="ICP Result")

    return pcd_moved, icp.transformation


def move_point_cloud(post_array, transform):
    """Takes in a nx3 ndarray of a point cloud, and a (4, 4) transform and transforms the point cloud

    Returns:
         post_array: The transformed point cloud (nx3 ndarray)
    """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(post_array)
    point_cloud.transform(transform)
    return np.asarray(point_cloud.points)


def compute_asd(pcd1, pcd2):
    """This function computes the average surface distance between two point clouds. The two point clouds are
    represented as nx3 ndarrays.

    Returns:
        asd: The average surface distance between the two point clouds
    """

    p_moved = o3d.geometry.PointCloud()
    p_moved.points = o3d.utility.Vector3dVector(pcd1)

    p_fixed = o3d.geometry.PointCloud()
    p_fixed.points = o3d.utility.Vector3dVector(pcd2)

    distances = p_moved.compute_point_cloud_distance(p_fixed)
    asd = np.mean(distances)
    return asd