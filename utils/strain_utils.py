## This file contains helper functions for performing strain and thickness analyses

import numpy as np
import open3d as o3d
import scipy
import pyvista as pv

def calculate_strain(pre_cart, pre_thick, post_cart, post_thick):
    """This function returns strain coordinates and strain values for two bone and thickness values"""

    # Find the closest points of the pre cartilage to the post cartilage
    distances = scipy.spatial.distance.cdist(post_cart, pre_cart)
    closest_indices = np.argmin(distances, axis=1)  # indices of the pre_cart closest to the post_cart

    # Initialize arrays
    strain_points = []
    strain = []

    # Iterate through the post cartilage points
    for i in range(len(post_cart)):
        # Get coordinates of the two corresponding points
        post_coord = post_cart[i]
        pre_coord = pre_cart[closest_indices[i]]

        # Find average thickness values within a 2.5 mm radius
        radius_mm = 2.5
        post_dists = np.linalg.norm(post_coord - post_cart, axis=1)
        post_inds = post_dists < radius_mm
        post_thick_here = np.mean(post_thick[post_inds])

        pre_dists = np.linalg.norm(pre_coord - pre_cart, axis=1)
        pre_inds = pre_dists < radius_mm
        pre_thick_here = np.mean(pre_thick[pre_inds])

        # Threshold distance. If the distance between these two coordinates isn't too large, add to strain map.
        dist_thresh = 1.0  # distance [mm] that signifies a "good" comparison

        if np.linalg.norm(pre_coord - post_coord) < dist_thresh:
            # Append the pre_coord to the strain map
            strain_points.append(pre_coord)

            # Calculate strain
            strain_here = (post_thick_here - pre_thick_here) / pre_thick_here
            strain.append(strain_here)

    # Convert coords and strain lists to numpy arrays
    strain_points = np.array(strain_points)
    strain = np.array(strain)

    return strain_points, strain


def average_point_cloud(points, values, radius=2.5):
    """This function averages the values associated with a point cloud (i.e. thickness or strain) based on a moving
    sphere with a specified radius.
    Inputs:
        points: Coordinates of the point cloud (nx3 ndarray)
        values: Thickness or strain values of this point cloud (nx1 ndarray)
        radius: Radius over which to average (scalar, same distance units as points)

    Returns:
          averaged_values: The averaged values for each point in the point cloud (nx3 ndarray)
    """

    averaged_values = np.zeros_like(values)

    for idx, coord in enumerate(points):
        distances = np.linalg.norm(coord - points, axis=1)  # Calculate distance to all other points
        inds = distances < radius  # Find all points within the radius
        thick_to_avg = values[inds]  # Get the thicknesses to average

        averaged_values[idx] = np.mean(thick_to_avg)

    return averaged_values


def remove_outer_boundaries(points, values, radius=5, n=7, percent=10):
    """This function removes the outer boundaries of a point cloud based on the density of points in the area. It
    searches over the entire point cloud, and removes points that are below a certain density threshold, as specified
    by the percent parameter.

    Inputs:
        points: Coordinates of the point cloud (nx3 ndarray)
        values: Thickness or strain values of this point cloud (nx1 ndarray)
        radius: Radius over which to average (scalar, same distance units as points)
        n: Number of iterations of removal to perform
        percent: Percentile of density to remove points below

    Returns:
        points_filtered: The filtered point cloud (nx3 ndarray)
        values_filtered: The filtered values (nx1 ndarray)
    """
    for i in range(n):
        # Estimate normals for the entire point cloud
        ptcld = o3d.geometry.PointCloud()
        ptcld.points = o3d.utility.Vector3dVector(points)

        # Estimate normals for point cloud
        ptcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        # Compute point density
        kdtree = o3d.geometry.KDTreeFlann(ptcld)
        densities = np.zeros(len(ptcld.points))
        for j, point in enumerate(ptcld.points):
            [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            densities[j] = len(idx)

        # Define density threshold (remove points below a specific density)
        density_threshold = np.percentile(densities, percent)

        # Filter points based on density
        points_filtered = []
        values_filtered = []
        for j, density in enumerate(densities):
            if density >= density_threshold:
                points_filtered.append(ptcld.points[j])
                values_filtered.append(values[j])

        # Overwrite points and values for next iteration
        points = np.array(points_filtered)
        values = np.array(values_filtered)

    # When done iterating...
    points_filtered = points
    values_filtered = values

    return points_filtered, values_filtered


def save_map(points, values, filename, text_to_add=None, colorbar_title="value", rng=None):
    """This function takes in a point map (thickness or strain), and saves it to a .png file. Optionally, you can at add text
    to the top of the image, or specify the range of the colormap

    Inputs:
        points: coordinates we're plotting (nx3 ndarray)
        values: values we're plotting, such as strain or thickness (nx1 ndarray)
        filename: The filename to save the map to (str)
        text_to_add: Text to add to the top of the image (str): Defaulting to None
        colorbar_title: Title of the colorbar (str): Defaulting to "value"
        rng: Range of the colormap (tuple). Defaulting to fit the data if none is specified
    """
    strain_cloud = pv.PolyData(np.transpose([points[:, 0], points[:, 1], points[:, 2]]))

    surf = strain_cloud.delaunay_2d(alpha=2)
    plotter = pv.Plotter(off_screen=True)
    surf[colorbar_title] = values

    if rng is None:
        rng = [np.min(values), np.max(values)]

    plotter.add_mesh(surf, show_edges=False, cmap="jet", rng=rng)

    # Set view vector: may need to change
    plotter.view_vector([0, -1, 0], [1, 0, 0])

    if text_to_add is not None:
        plotter.add_text(text_to_add, font_size=10, color="black", position="upper_edge", font="times")

    plotter.screenshot(filename)
    plotter.close()
    return


def calculate_thickness(bone_points, cart_points):
    """This function calculates the thickness for every point in the cart_points array.

    Inputs:
        bone_points: The bone point cloud (nx3 ndarray)
        cart_points: The cartilage point cloud (nx3 ndarray)
    Returns:
        thickness: The thickness of the cartilage at each point (nx1 ndarray)"""

    distances = scipy.spatial.distance.cdist(cart_points, bone_points)
    thickness = np.min(distances, axis=1)
    return thickness

