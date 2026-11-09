import numpy as np
import networkx as nx
try:
    import triangle as tr
except ImportError:  # only needed to generate D-Hydro meshes (create_mesh_triangle, Mesh._import_from_Triangle)
    tr = None
import os
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from typing import List, Tuple
import pickle
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.linalg import lstsq
from scipy.interpolate import griddata
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.utils import scatter
from shapely.geometry import Polygon
import xarray as xr
try:
    from meshkernel import MeshKernel, Mesh2d, GeometryList, OrthogonalizationParameters, ProjectToLandBoundaryOption, MeshRefinementParameters
    from meshkernel import py_structures, DeleteMeshOption
except ImportError:  # only needed to generate D-Hydro meshes (create_mesh_dhydro, Mesh._import_from_meshkernel)
    MeshKernel = Mesh2d = GeometryList = OrthogonalizationParameters = None
    ProjectToLandBoundaryOption = MeshRefinementParameters = py_structures = DeleteMeshOption = None

def center_grid_graph(dim1, dim2, grid_size=1):
    '''
    Create graph from a rectangular grid of dimensions dim1 x dim2
    Returns networkx graph connecting the grid centers and corresponding 
    node positions
    ------
    dim1: int
        number of grids in the x direction
    dim2: int
        number of grids in the y direction
    '''
    G = nx.grid_2d_graph(dim1, dim2, create_using=nx.DiGraph)
    # for the position, it is assumed that they are located in the centre of each grid
    pos = {i:((x+0.5)*grid_size,(y+0.5)*grid_size) for i, (x,y) in enumerate(G.nodes())}
    
    #change keys from (x,y) format to i format
    mapping = dict(zip(G, range(0, G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    return G, pos

def get_coords(pos):
    '''
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    return np.array([xy for xy in pos.values()])

def mesh_radius_graph(face_xy, max_radius=100):
    radius_graph = radius_neighbors_graph(face_xy, max_radius, mode='connectivity', include_self=False)
    new_edge_index = np.stack(radius_graph.nonzero())
    assert new_edge_index.shape[1] > 0, 'There are no edges with the selected radius, try increasing it'

    return new_edge_index

def dual_graph_from_mesh(mesh):
    graph = nx.from_edgelist(mesh.dual_edge_index.T)
    pos = mesh.face_xy
    return graph, pos

def graph_from_mesh(mesh):
    graph = nx.from_edgelist(mesh.edge_index.T)
    pos = mesh.node_xy
    return graph, pos

def sample_points_from_grid(points, grid_size):
    """Sample points from a regular square grid. 
    Each square in the grid will have a single point selected at random.
    
    Args:
        points (np.array): Array of points to sample from, shape (n_points, 2)
        grid_size (float): Size of each square in the grid
    """
    num_points = 1  # Number of points to select in each square

    # Generate the regular grid
    x = np.arange(points[:,0].min(), points[:,0].max() + grid_size, grid_size)
    y = np.arange(points[:,1].min(), points[:,1].max() + grid_size, grid_size)

    selected_points = []
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            # Create a mask for each square
            square_mask = ((points[:,0] > x[i]) & (points[:,0] < x[i+1])) & \
                          ((points[:,1] > y[j]) & (points[:,1] < y[j+1]))
            possible_points = points[square_mask]

            if len(possible_points) > 0:
                # Generate random indices within the square
                indices = np.random.randint(0, len(possible_points), size=min(num_points, len(possible_points)))
                
                # Add the selected points to the list
                selected_points.extend(possible_points[indices])

    selected_points = np.array(selected_points)

    print(f"Sampled {(selected_points.shape[0] / points.shape[0])*100:0.1f}% of the nodes")

    return selected_points

def get_ordered_boundary_nodes(mesh):
    # determine boundary nodes in mesh
    boundary_nodes = mesh.boundary_nodes

    # create KNN graph from the boundary nodes
    knn_graph = kneighbors_graph(boundary_nodes, 2, mode='distance', include_self=False)
    knn_graph = knn_graph + knn_graph.T

    # get edges from knn graph
    boundary_edge_index = np.array(knn_graph.nonzero()).T
    edges = np.unique(np.sort(boundary_edge_index, axis=1), axis=0)

    ordered_nodes = [edges[0,0], edges[0,1]]
    edges = np.delete(edges, 0, axis=0)

    for i in range(len(edges)-1):
        row, index = np.where(edges == ordered_nodes[-1])
        ordered_nodes.append(edges[row, index-1].item())
        edges = np.delete(edges, row, axis=0)

    ordered_nodes = np.array(ordered_nodes)

    return boundary_nodes[ordered_nodes], boundary_edge_index

def get_boundary_corners(boundary_nodes):
    boundary_nodes = np.vstack((boundary_nodes, boundary_nodes[:2]))
    removable_nodes = []
    for i in range(len(boundary_nodes)):
        if get_polygon_area(boundary_nodes[i:i+3, 0], boundary_nodes[i:i+3, 1]) == 0:
            removable_nodes.append(i)

    boundary_corners = np.delete(boundary_nodes, removable_nodes, axis=0)

    return boundary_corners

def close_polygon(points):
    if not np.all(points[0] == points[-1]):
        points = np.vstack((points, points[0]))
    return points

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float, 
                     num_vertices: int, seed: float, ellipticality: float=1) -> List[Tuple[float, float]]:
    """
    TAKEN FROM: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to the center of the circumference) 
            used to generate points with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of the circumference.
        ellipticality (float):
            ratio between the major and minor axis of the ellipse used to generate the polygon.            
        num_vertices (int):
            the number of vertices of the polygon.
    """
    np.random.seed(seed)

    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity, seed)

    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        radius = clip(np.random.normal(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * np.cos(angle) * ellipticality,
                 center[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    polygon = Polygon(points)

    return polygon

def random_angle_steps(steps: int, irregularity: float, seed: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    np.random.seed(seed)

    # generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def equidistant_perimiter(vertices):
    segments_lengths = np.array([np.linalg.norm(vertices[i+1,:] - vertices[i,:]) for i in range(len(vertices)-1)])
    min_length = segments_lengths.min()

    new_vertices = copy(vertices)
    for i in range(len(vertices)-1):
        segment_ratio = segments_lengths[i] / min_length
        if segment_ratio > 2:
            more_segments = np.linspace(vertices[i,:], vertices[i+1,:], int(np.ceil(segment_ratio/2)))
            index = i + len(new_vertices) - len(vertices)
            new_vertices = np.concatenate((new_vertices[:index+1,:], more_segments[1:-1], new_vertices[index+1:,:]))

    return new_vertices

def save_polygon_to_file(polygon, filename):
    """Save the generated polygon to a .pol file."""
    with open(filename, 'w') as f:
        f.write(f'# Extent: {polygon.bounds}\n')
        f.write('# Coordinates (x, y):\n')
        for coord in polygon.exterior.coords:
            f.write(f'{coord[0]}, {coord[1]}\n')

def create_dike(dike_corners:list, dike_points_frequency:int=1, dike_width:float=1):
    """Returns an array of points that discretize a linear element defined by its corners"""
    x0, y0 = dike_corners[0]
    x1, y1 = dike_corners[1]

    if y1-y0 > x1-x0:
        if y0 > y1:
            dike_points_frequency *= -1
        y = np.arange(y0, y1, dike_points_frequency)
        x = np.linspace(x0, x1, y.shape[0])

        diagonal1 = np.stack((x, y), 1)
        diagonal2 = np.stack((x+dike_width, y), 1)

    else:
        if x0 > x1:
            dike_points_frequency *= -1
        x = np.arange(x0, x1, dike_points_frequency)
        y = np.linspace(y0, y1, x.shape[0])

        diagonal1 = np.stack((x, y), 1)
        diagonal2 = np.stack((x, y+dike_width), 1)

    return np.concatenate((diagonal1, diagonal2))

def is_point_inside_polygon(point, polygon):
    '''Determine if a point (x,y) is inside a polygon (list of points)'''
    x, y = point
    n = len(polygon)

    for i in range(n):
        inside = False
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 < y and y2 >= y or y2 < y and y1 >= y:
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) > x:
                inside = True
                break

    return inside

def check_coarsening(points, boundary_nodes):
    '''Determine if an array of points (x,y) is inside a polygon (list of points)'''
    # import path
    from matplotlib.path import Path

    # check if the points are inside the convex hull
    hull_path = Path(boundary_nodes)
    inside = hull_path.contains_points(points)

    if (~inside).sum() > 0:
        print(f"{(~inside).sum()} points are outside or on the convex hull")

    return inside

def generate_random_polygon_with_dike(save_polygon=False, avg_radius=100, irregularity=0.5, 
                                      spikiness=0.2, seed=42, num_vertices=(20,30), ellipticality=(1,2),
                                      dike_corners=None, min_dike_length=0.5, **dike_options):
    """Generates a polygon with inside it a linear dike element"""
    np.random.seed(seed)

    num_vertices = np.random.randint(num_vertices[0], num_vertices[1])
    ellipticality = np.random.uniform(ellipticality[0], ellipticality[1])
    avg_radius = avg_radius/ellipticality
    polygon = generate_polygon(center=(avg_radius, avg_radius), avg_radius=avg_radius, 
                               irregularity=irregularity, spikiness=spikiness,
                               ellipticality=ellipticality,
                               num_vertices=num_vertices, seed=seed)

    if save_polygon:
        save_polygon_to_file(polygon, 'random_polygon.pol')

    vertices = np.array(polygon.exterior.coords)
    vertices = equidistant_perimiter(vertices)
    
    if dike_corners is None:
        dike_length = 0
        while dike_length < min_dike_length:
            dike_corners = np.random.random((2,2))
            dike_length = np.linalg.norm((dike_corners[1]-dike_corners[0]))
        dike_corners *= avg_radius*1.5

    # dike = create_dike(dike_corners, **dike_options)
    # outside_points = np.array([is_point_inside_polygon(point, vertices) for point in dike])
    # vertices, dike[outside_points]

    return None

def plot_faces(mesh, ax=None, face_value=None, **kwargs):
    """Plots the mesh with face values if specified"""
    ax = ax or plt.gca()

    node_position = 0
    patches = []
    for num_nodes in mesh.nodes_per_face:
        face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh.node_x[face_node]
        face_nodes_y = mesh.node_y[face_node]
        face = np.stack((face_nodes_x, face_nodes_y)).T
        node_position += num_nodes
        patches.append(mpl.patches.Polygon(face, closed=True))
        
    collection = PatchCollection(patches, **kwargs)
    collection.set_array(face_value)
    ax.add_collection(collection)
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax

def plot_mesh(mesh, ax=None, node_size=2, **plt_kwargs):
    ax = ax or plt.gca()
    
    graph, pos = graph_from_mesh(mesh)

    nx.draw(graph, pos, ax=ax, node_size=node_size, **plt_kwargs)

    return ax

def plot_mesh_and_dual(mesh, ax=None, **plt_kwargs):
    ax = ax or plt.gca()
    
    graph, pos = graph_from_mesh(mesh)

    nx.draw(graph, pos, ax=ax, style='dotted', node_size=0, width=0.2)

    dual_graph, pos = dual_graph_from_mesh(mesh)
    
    node_size = 4000/dual_graph.number_of_nodes()**0.9
    nx.draw(dual_graph, pos, ax=ax, node_size=node_size, width=0.7, **plt_kwargs)
    
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax

def plot_multiscale_mesh_properties(meshes, with_area=True, **kwargs):
    """Plot the mesh properties of a multiscale mesh.
    
    ----------
    meshes : list of Mesh objects
    with_area : bool, plot the histogram of the face areas"""
    number_of_multiscales = len(meshes)
    
    height_ratios = [1, 0.65] if with_area else [1]

    fig, axs = plt.subplots(1+with_area, number_of_multiscales, figsize=(number_of_multiscales*4,4+with_area*2), 
                            gridspec_kw={'height_ratios': height_ratios})
    fig.suptitle("Mesh faces properties", fontsize=16)

    for i, mesh in enumerate(meshes):

        if with_area:
            ax0 = axs[0, i] if number_of_multiscales > 1 else axs[0]
            ax1 = axs[1, i] if number_of_multiscales > 1 else axs[1]
            ax1.hist(mesh.face_area, bins=30)
            ax1.set_xlabel("Face Area")
        else:
            ax0 = axs[i] if number_of_multiscales > 1 else axs
        plot_mesh(mesh, ax=ax0, **kwargs)
        ax0.set_title(f"Num faces: {mesh.face_x.shape[0]}")

    plt.show()

def connect_coarse_to_fine_mesh(coarse_mesh, fine_mesh):
    """Connects the coarse mesh to the fine mesh by creating a dual edge index between the two meshes.
    An edge is created if the center of the fine mesh face is contained in the coarse mesh face."""
    assert isinstance(coarse_mesh, Mesh) and isinstance(fine_mesh, Mesh), "coarse_mesh and fine_mesh must be Mesh objects"

    coarse_face_nodes = get_face_nodes_mesh(coarse_mesh)
    fine_face_xy = fine_mesh.face_xy

    coarse_polygons = [Path(coarse_mesh.node_xy[face_nodes[~np.isnan(face_nodes)].astype(int)]) for face_nodes in coarse_face_nodes]
    contained_faces = [polygon.contains_points(fine_face_xy) for polygon in coarse_polygons]

    coarse_to_fine_dual_edge_index = np.column_stack(np.where(contained_faces))
    # fine_to_coarse_dual_edge_index = coarse_to_fine_dual_edge_index[:, ::-1]

    return coarse_to_fine_dual_edge_index

def get_barycenter(x, y):
    '''Returns barycenter given x and y coordinates'''
    assert x.shape == y.shape, f"Input x and y have incompatible dimensions \n\
                                x: {x.shape}, y: {y.shape}"
    
    if x.ndim == 1:
        length = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
    elif x.ndim == 2:
        length = x.shape[1]
        sum_x = np.sum(x, 1)
        sum_y = np.sum(y, 1)
    else:
        raise ValueError("The dimension of the arrays is wrong")

    return sum_x/length, sum_y/length

def create_mesh_triangle(vertices, segments=None, holes=None, max_area=5, max_smallest_mesh_angle=30):
    '''Creates a mesh using Triangle'''

    if max_smallest_mesh_angle>34:
        raise ValueError("Mesh not computed. Triangle doesn't like hard restrictions.")

    mesh_inputs = {'vertices': vertices}
    
    if segments is not None:
        mesh_inputs['segments'] = segments
    if holes is not None:
        mesh_inputs['holes'] = holes
        
    mesh = tr.triangulate(mesh_inputs, f'cq{max_smallest_mesh_angle}pena{max_area}iD')

    return mesh

def create_mesh_dhydro(polygon_file='random_polygon.pol', number_of_multiscales=4,
                       for_simulation=True):
    '''Creates a fine mesh or a multiscale mesh using meshkernel
    
    ------
    polygon_file: str, path-like
        path to the polygon file
    number_of_multiscales: int
        number of multiscale levels/number of refinement operations
    for_simulation: bool
        if True, returns the fine mesh, otherwise returns the multiscale mesh
    '''
    # mesh = create_mesh_triangle(vertices, segments=None, holes=None, max_area=max_area, max_smallest_mesh_angle=30)
    # mesh2d = Mesh2d(node_x=np.array(mesh['vertices'][:,0], dtype=np.float64),
    #             node_y=np.array(mesh['vertices'][:,1], dtype=np.float64),
    #             edge_nodes=mesh['edges'].ravel())
    # mk = MeshKernel()
    # mk.mesh2d_set(mesh2d)

    with open(polygon_file) as file:
        boundary_nodes = np.array([[value for value in line.strip().split(",")] for line in file.readlines()[2:]], dtype=np.double)

    boundary_polygon = GeometryList(boundary_nodes[:,0].copy(), boundary_nodes[:,1].copy())
    meshes = []

    mk = MeshKernel()
    mk.mesh2d_make_triangular_mesh_from_polygon(boundary_polygon)

    for i in range(number_of_multiscales):
        mk.mesh2d_compute_orthogonalization(ProjectToLandBoundaryOption(0), OrthogonalizationParameters(
                    outer_iterations=25, boundary_iterations=25, inner_iterations=25, 
                    orthogonalization_to_smoothing_factor=0.975),
                    boundary_polygon, boundary_polygon)
        
        if i == number_of_multiscales-1:
            mk.mesh2d_delete_small_flow_edges_and_small_triangles(
            small_flow_edges_length_threshold=0.1, min_fractional_area_triangles=2.0)
            
        mesh = Mesh()
        mesh._import_from_meshkernel(mk)
        meshes.append(mesh)

        if i < number_of_multiscales-1:
            refinement_parameters = MeshRefinementParameters(refine_intersected=True, min_edge_size=0.5, 
                                                        max_refinement_iterations=1, smoothing_iterations=5)
            mk.mesh2d_refine_based_on_polygon(boundary_polygon, refinement_parameters)
        
    if for_simulation:
        output_mesh2d = mk.mesh2d_get()

        output_mesh2d.mesh_nodes = np.stack((output_mesh2d.node_x, output_mesh2d.node_y), -1)
        output_mesh2d.face_xy = np.stack((output_mesh2d.face_x, output_mesh2d.face_y), -1)

        return output_mesh2d
    else:
        return meshes

def get_face_nodes_mesh(mesh):
    num_faces = mesh.face_x.shape[0]
    max_nodes_per_face = mesh.nodes_per_face.max()
    face_nodes = np.zeros((num_faces, max_nodes_per_face)) * np.nan
    node_position = 0

    for i, num_nodes in enumerate(mesh.nodes_per_face):
        face_nodes[i,:num_nodes] = mesh.face_nodes[node_position : (node_position + num_nodes)]
        node_position += num_nodes

    return face_nodes

def save_mesh(mesh, mesh_file):
    '''Saves mesh as NETCDF file using ugrid
    ------
    mesh: meshkernel.py_structures.Mesh2d
    mesh_file: str, path-like
        path to output file
    '''
    # from ugrid import UGrid

    # with UGrid(mesh_file, "w+") as ug:
    #     mesh.node_x = mesh.node_x*grid_size
    #     mesh.node_y = mesh.node_y*grid_size

    #     # 1. Convert a meshkernel mesh2d to an ugrid mesh2d
    #     mesh2d_ugrid = ug.from_meshkernel_mesh2d_to_ugrid_mesh2d(mesh2d=mesh, name="Mesh2d", is_spherical=False)

    #     # 2. Define a new mesh2d
    #     topology_id = ug.mesh2d_define(mesh2d_ugrid)

    #     # 3. Put a new mesh2d
    #     ug.mesh2d_put(topology_id, mesh2d_ugrid)

    #     # 4. Add crs to file
    #     attribute_dict = {
    #         "name": "Unknown projected",
    #         "epsg": np.array([0], dtype=int),
    #         "grid_mapping_name": "Unknown projected",
    #         "longitude_of_prime_meridian": np.array([0.0], dtype=float),
    #         "semi_major_axis": np.array([6378137.0], dtype=float),
    #         "semi_minor_axis": np.array([6356752.314245], dtype=float),
    #         "inverse_flattening": np.array([6356752.314245], dtype=float),
    #         "EPSG_code": "EPSG:0",
    #         "value": "value is equal to EPSG code"}
    #     ug.variable_int_with_attributes_define("projected_coordinate_system", attribute_dict)

    from netCDF4 import Dataset

    if os.path.exists(mesh_file): os.remove(mesh_file)

    with Dataset(mesh_file, mode='w', format='NETCDF4') as ncfile:
        ncfile.createDimension('nNetNode', mesh.node_x.shape[0])
        ncfile.createDimension('nNetLink', mesh.edge_nodes.shape[0]//2)
        ncfile.createDimension('nNetLinkPts', 2)

        ncfile.createVariable('projected_coordinate_system', 'int32', ())
        NetNode_x = ncfile.createVariable('NetNode_x', 'f8', ('nNetNode',))
        NetNode_y = ncfile.createVariable('NetNode_y', 'f8', ('nNetNode',))
        NetNode_z = ncfile.createVariable('NetNode_z', 'f8', ('nNetNode',))
        NetLink = ncfile.createVariable('NetLink', 'int32', ('nNetLink', 'nNetLinkPts'))
        NetLinkType = ncfile.createVariable('NetLinkType', 'int32', ('nNetLink'))

        NetNode_x.units = "m"
        NetNode_x.long_name = "x-coordinate"
        NetNode_y.units = "m"
        NetNode_y.long_name = "y-coordinate"
        NetNode_x.coordinates = "NetNode_x NetNode_y"
        NetNode_y.coordinates = "NetNode_x NetNode_y"
        NetNode_z.coordinates = "NetNode_x NetNode_y"

        NetNode_x[:] = mesh.node_x
        NetNode_y[:] = mesh.node_y
        NetNode_z[:] = np.zeros_like(mesh.node_y)
        NetLink[:] = mesh.edge_nodes.reshape(-1,2)+1
        NetLinkType[:] = np.ones(mesh.edge_nodes.shape[0]//2, dtype=np.int32)-1

def get_polygon_area(x, y):
    '''Apply shoelace algorithm to evaluate area defined by sequence of points (x,y)'''
    assert x.shape == y.shape, f"Input x and y have incompatible dimensions \n\
                                x: {x.shape}, y: {y.shape}"
    if x.ndim == 1:
        area = 0.5*np.abs(np.dot(x,np.roll(y,1,axis=-1))
            -np.dot(y,np.roll(x,1,axis=-1)))
    elif x.ndim == 2:
        area = 0.5*np.abs(np.multiply(x,np.roll(y,1,axis=-1)).sum(1)
            -np.multiply(y,np.roll(x,1,axis=-1)).sum(1))
    else:
        raise ValueError(f"Input x and y have incorrect dimension ({x.shape})")
    return area

class Mesh(object):
    def __init__(self):
        '''Mixed-elements mesh base object
        ------
        node_x: np.array, shape (num_nodes,)
            x coordinates of each node
        node_y: np.array, shape (num_nodes,)
            y coordinates of each node
        face_x: np.array, shape (num_faces,)
            x coordinates of each face
        face_y: np.array, shape (num_faces,)
            y coordinates of each face
        edge_index: np.array, shape (2, num_edges)
            index of connected nodes
        edge_type: np.array, shape (num_edges,)
            type of each edge (1:normal edges, 2:edge with boundary condition, 3:other boundary edges)
        dual_edge_index: np.array, shape (2, num_dual_edges)
            index of connected faces
        face_nodes: np.array, shape (num_faces*nodes_per_face,)
            index of the nodes that define each face
        nodes_per_face: np.array, shape (num_faces,)
            number of nodes that define each face
        '''
        self.added_ghost_cells = False
        self.node_x = np.array([])
        self.face_x = np.array([])
        self.edge_index = np.array([[]])
        self.dual_edge_index = np.array([[]])

    def _import_from_map_netcdf(self, nc_file):
        """Import mesh from map netcdf file (the output of DHYDRO)

        -------
        Adds the following attributes:
        edge_index_BC: np.array, shape (num_edges_BC, 2)
            index of the edges that have boundary conditions
        face_BC: np.array, shape (num_faces_BC,)
            index of the faces that have boundary conditions
        edge_BC: np.array, shape (num_edges_BC,)
            index of the edges that have boundary conditions
        extra_face_BC: np.array, shape (num_extra_faces_BC,)
            index of the faces that are in the boundary but with no boundary conditions
        
        -------
        Updates the following attributes:
        dual_edge_index: removes the edges that are in the boundary but with no boundary conditions
        """
        nc_dataset = xr.open_dataset(nc_file)
        self.node_x = nc_dataset['mesh2d_node_x'].data
        self.node_y = nc_dataset['mesh2d_node_y'].data

        self.face_x = nc_dataset['mesh2d_face_x'].data
        self.face_y = nc_dataset['mesh2d_face_y'].data

        self.edge_index = nc_dataset['mesh2d_edge_nodes'].data.T - 1
        self.edge_type = nc_dataset['mesh2d_edge_type'].data # 1:normal edges, 2:BC_edge, 3:other boundary edges
        self.dual_edge_index = nc_dataset['mesh2d_edge_faces'].data.T.astype(int) - 1

        self.face_nodes = nc_dataset['mesh2d_face_nodes'] - 1
         # mixed mesh
        if isinstance(self.face_nodes.to_masked_array().mask, np.ndarray):
            self.nodes_per_face = (~self.face_nodes.to_masked_array().mask).sum(1).astype(int)
            self.face_nodes = self.face_nodes.data[~self.face_nodes.to_masked_array().mask].astype(int)
        # triangular or quadrilateral mesh
        else:
            self.nodes_per_face = np.ones_like(self.face_nodes).sum(1).data.astype(int)
            self.face_nodes = self.face_nodes.reshape(-1).data.astype(int)

        self.edge_index_BC = self.edge_index[:,self.edge_type == 2].T
        self.boundary_edges = self.edge_index[:,self.edge_type > 1].T
        self.edge_BC = np.stack([np.where((edge==self.edge_index.T).sum(1) == 2) for edge in self.edge_index_BC]).reshape(-1)

        face_bnd_mask = self.dual_edge_index[0,:] == -1
        self.face_BC = self.dual_edge_index[1,face_bnd_mask]

        extra_face_bnd_mask = self.dual_edge_index[1,:] == -1
        self.face_bnd = self.dual_edge_index[0,extra_face_bnd_mask]

        total_face_bnd_mask = extra_face_bnd_mask | face_bnd_mask
        self.dual_edge_index = self.dual_edge_index[:,~total_face_bnd_mask]
        self.dual_edge_index = to_undirected(torch.LongTensor(self.dual_edge_index)).numpy() #convert to undirected graph
        self._get_derived_attributes()

    def _import_from_meshkernel(self, meshkernel_mesh):
        """Import mesh from meshkernel Mesh2d object
        
        Example to create a Mesh2d object:
        mk = MeshKernel()
        mk.mesh2d_make_rectilinear_mesh(0, 0, 100, 100, 10, 10)
        
        mesh = Mesh()
        mesh._import_from_meshkernel(mk)
        """
        assert isinstance(meshkernel_mesh, MeshKernel), 'Input mesh must be a MeshKernel object from meshkernel'
        mesh = meshkernel_mesh.mesh2d_get()

        self.node_x = mesh.node_x
        self.node_y = mesh.node_y
        self.node_xy = np.stack((self.node_x, self.node_y),-1)
        
        self.face_x = mesh.face_x
        self.face_y = mesh.face_y

        self.edge_index = mesh.edge_nodes.reshape(-1,2).T
        self.dual_edge_index = mesh.edge_faces.reshape(-1,2).T
        
        extra_face_bnd_mask = self.dual_edge_index[1,:] == -1
        self.face_bnd = self.dual_edge_index[0,extra_face_bnd_mask]
        self.dual_edge_index = self.dual_edge_index[:,~extra_face_bnd_mask]
        self.dual_edge_index = to_undirected(torch.LongTensor(self.dual_edge_index)).numpy() #convert to undirected graph

        self.face_nodes = mesh.face_nodes
        self.nodes_per_face = mesh.nodes_per_face

        boundary_polygon = meshkernel_mesh.mesh2d_get_mesh_boundaries_as_polygons()
        self.boundary_nodes = np.stack((boundary_polygon.x_coordinates, boundary_polygon.y_coordinates),-1)
        boundary_nodes_ids = [np.where((self.node_xy == node).all(1))[0][0] for node in self.boundary_nodes[:-1]]
        boundary_edge = np.stack([boundary_nodes_ids[i:i+2] for i in range(len(boundary_nodes_ids)-1)]).T
        # boundary_nodes_ids = np.array([i for i in range(len(self.boundary_nodes)-1)])
        # boundary_edge = np.array([[i, (i+1)%len(boundary_nodes_ids)] for i in boundary_nodes_ids]).T
        boundary_edge_id = np.array([np.where((self.edge_index.T == edge).all(1) | 
                                              (self.edge_index[::-1].T == edge).all(1))[0] 
                                              for edge in boundary_edge.T]).squeeze()
        self.edge_type = np.ones(self.edge_index.shape[1])
        self.edge_type[boundary_edge_id] = 3
        self.boundary_edges = self.edge_index[:,self.edge_type > 1].T
        self._get_derived_attributes()

    def _import_from_Triangle(self, mesh):
        """Import mesh from triangle mesh object.
        The triangulation must have -en flags activated
        
        Example:
        mesh_options = {"vertices": points}
        mesh = tr.triangulate(mesh_options, 'en')
        """
        self.edge_index = mesh['edges'].T
        self.edge_type = mesh['edge_markers'].squeeze() * 2 + 1
        self.node_x = mesh['vertices'][:, 0]
        self.node_y = mesh['vertices'][:, 1]
        self.face_nodes = mesh['triangles'].ravel()

        self.face_x = self.node_x[mesh['triangles']]  # shape [F, 3]
        self.face_y = self.node_y[mesh['triangles']]  # shape [F, 3]

        self.nodes_per_face = np.ones(len(self.face_x), dtype=int) * 3  # times 3 because it's a triangle

        dual_edge_index = np.array([[[face, neighbour]
                                          for neighbour in mesh['neighbors'][face]]
                                         for face in range(len(self.face_x))]
                                        ).reshape(-1, 2).T  # shape [2, E_d]
        self.dual_edge_index = to_undirected(torch.LongTensor(dual_edge_index)).numpy() #convert to undirected graph
        self.boundary_nodes = mesh['vertex_markers']

    def _get_derived_attributes(self):
        """Calculate derived attributes from the mesh
        ------
        node_xy: np.array
            x and y coordinates of each node
        num_nodes: int
            number of nodes in the mesh
        boundary_nodes: np.array
            x and y coordinates of each boundary node
        edge_relative_distance: np.array
            relative distance between the nodes that define each edge
        edge_length: np.array
            length of each edge
        edge_outward_normal: np.array
            outward normal of each edge
        num_edges: int
            number of edges in the mesh
        face_xy: np.array
            x and y coordinates of each face
        face_relative_distance: np.array
            relative distance between the faces that define each edge
        dual_edge_length: np.array
            length of each dual edge
        num_faces: int
            number of faces in the mesh
        face_area: np.array 
            area of each face
        """
        # Nodes
        self.node_xy = np.stack((self.node_x, self.node_y),-1)
        self.num_nodes = self.node_x.shape[0]
        self.boundary_nodes = self.node_xy[np.array(list(set(self.edge_index.T[self.edge_type > 1].flatten())))]
        
        # Edges        
        self.edge_relative_distance = self.node_xy[self.edge_index[1,:]] - self.node_xy[self.edge_index[0,:]]
        self.edge_length = np.linalg.norm(self.edge_relative_distance, axis=1)

        self.edge_outward_normal = self.edge_relative_distance/self.edge_length[:,None]
        self.edge_outward_normal[:,1] = -self.edge_outward_normal[:,1]
        self.num_edges = self.edge_index.shape[1]

        # Faces
        self.face_xy = np.stack((self.face_x, self.face_y),-1)
        self.face_relative_distance = self.face_xy[self.dual_edge_index[1,:]] - self.face_xy[self.dual_edge_index[0,:]]
        self.dual_edge_length = np.linalg.norm(self.face_relative_distance, axis=1)
        self.num_faces = self.face_x.shape[0]

        node_position = 0
        face_areas = []
        for num_nodes in self.nodes_per_face:
            face_node = self.face_nodes[node_position : (node_position + num_nodes)]
            face_nodes_x = self.node_x[face_node]
            face_nodes_y = self.node_y[face_node]
            node_position += num_nodes
            face_area = get_polygon_area(face_nodes_x, face_nodes_y)
            if face_area == 0: raise ValueError(f"Face {face_node} has area equal to zero")
            face_areas.append(face_area)
        self.face_area = np.array(face_areas)

    def _import_DEM(self, DEM_file):
        """Import DEM file and interpolate it on the mesh
        ------
        DEM_file: str, path-like
            path to DEM file. It must be a file with three columns: x, y, z
        """
        try:
            DEM = np.loadtxt(DEM_file)
            self.DEM = interpolate_variable(self.face_xy, DEM[:,:2], DEM[:,2], method='nearest')
        except FileNotFoundError:
            print(f"Could not find the DEM file {DEM_file}. Setting DEM to zeros.")
            self.DEM = np.zeros_like(self.face_area)

    def plot_boundary(self, ax=None, **plt_kwargs):
        '''Plot the boundary of the mesh'''
        ax = ax or plt.gca()

        boundary_edges = np.concatenate((self.boundary_edges, np.array([self.boundary_edges[0,-1], self.boundary_edges[0,0]]).reshape(1,-1)))
        [ax.plot(self.node_xy[edge][:,0], self.node_xy[edge][:,1], **plt_kwargs) for edge in boundary_edges];

        return ax

    def __repr__(self) -> str:
        return 'Mesh object with {} nodes, {} edges, {} faces, and {} dual edges'.format(
            self.node_x.shape[0], self.edge_index.shape[1], self.face_x.shape[0], self.dual_edge_index.shape[1])
    
class MultiscaleMesh(Mesh):
    """Mesh class for multiscale meshes"""
    def __init__(self):
        super().__init__()
        self.num_meshes = 0

    def stack_meshes(self, meshes):
        self.num_meshes = len(meshes)
        self.meshes = meshes

        # stack node attributes
        self.node_x = np.concatenate([mesh.node_x for mesh in meshes])
        self.node_y = np.concatenate([mesh.node_y for mesh in meshes])

        # stack face attributes
        self.face_x = np.concatenate([mesh.face_x for mesh in meshes])
        self.face_y = np.concatenate([mesh.face_y for mesh in meshes])
        self.nodes_per_face = np.concatenate([mesh.nodes_per_face for mesh in meshes])

        # stack edge attributes
        self.edge_type = np.concatenate([mesh.edge_type for mesh in meshes])

        # stack edge indexes
        edge_index = [meshes[0].edge_index]
        dual_edge_index = [meshes[0].dual_edge_index]
        face_nodes = [meshes[0].face_nodes]

        for i, mesh in enumerate(meshes[1:]):
            edge_index.append(mesh.edge_index + edge_index[i].max() + 1)
            dual_edge_index.append(mesh.dual_edge_index + dual_edge_index[i].max() + 1)
            face_nodes.append(mesh.face_nodes + face_nodes[i].max() + 1)
            
        self.edge_index = np.concatenate(edge_index, 1)
        self.dual_edge_index = np.concatenate(dual_edge_index, 1)
        self.face_nodes = np.concatenate(face_nodes)
        
        # partition and compose the meshes
        self.get_partitioning(meshes)
        self.get_multiscale_BC(meshes)
        self.get_intra_edges(meshes)

        # adding derived attributes and ghost cells
        self._get_derived_attributes()    
        self.added_ghost_cells = True
        self.ghost_cells_ids = np.concatenate([mesh.ghost_cells_ids + self.face_ptr[i] for i, mesh in enumerate(meshes)])
        self.ghost_node_ids = [self.node_x.shape[0]-j-1 for j in range((self.nodes_per_face[self.face_BC]-2).sum())][::-1]
        
        dual_edge_index_BC = [meshes[0].dual_edge_index_BC]
        for i, mesh in enumerate(meshes[1:]):
            dual_edge_index_BC.append(mesh.dual_edge_index_BC + dual_edge_index_BC[i].max() + 1)
        self.dual_edge_index_BC = np.concatenate(dual_edge_index_BC, 1)

    def get_intra_edges(self, meshes, add_edges=False):
        """Adds dual edges across each multiscale level
        based on the position of the fine mesh centers in the coarse mesh
        
        Adds:
            intra_mesh_dual_edge_index (np.array): dual edge index of the intra mesh edges

        Updates:
            dual_edge_index: adds the intra mesh dual edges
        """
        if meshes[0].num_nodes < meshes[1].num_nodes:   # coarse to fine
            intra_mesh_dual_edge_index = [connect_coarse_to_fine_mesh(meshes[i], meshes[i+1]) + [self.face_ptr[i], self.face_ptr[i+1]] for i in range(len(meshes)-1)]
        else:   # fine to coarse
            intra_mesh_dual_edge_index = [connect_coarse_to_fine_mesh(meshes[i+1], meshes[i]) + [self.face_ptr[i+1], self.face_ptr[i]] for i in range(len(meshes)-1)]
        self.intra_edge_ptr = np.cumsum([0] + [edge.shape[0] for edge in intra_mesh_dual_edge_index])
        self.intra_mesh_dual_edge_index = np.concatenate(intra_mesh_dual_edge_index).T

        if add_edges:
            self.with_intra_edges = True
            self.dual_edge_index = np.concatenate([self.dual_edge_index, self.intra_mesh_dual_edge_index], 1)

    def remove_intra_edges(self):
        """Removes the intra mesh dual edges"""
        if self.with_intra_edges:
            self.with_intra_edges = False
            if self.added_ghost_cells:
                self.dual_edge_index = np.concatenate((self.dual_edge_index[:,:-self.intra_mesh_dual_edge_index.shape[1]-len(self.face_BC)], 
                                                    self.dual_edge_index_BC), 1)
            else:
                self.dual_edge_index = self.dual_edge_index[:,:-self.intra_mesh_dual_edge_index.shape[1]]
        else:
            print("The mesh does not have intra mesh dual edges. You can add them with add_intra_edges(meshes)")

    def get_multiscale_BC(self, meshes):
        """Get the boundary conditions for the multiscale mesh by 
        stacking the boundary conditions of the meshes
        
        Adds:
            edge_BC: index of boundary edges
            face_BC: index of boundary faces
            edge_index_BC: index of boundary edges in edge_index
        """
        if not all([hasattr(mesh, 'edge_BC') for mesh in meshes]):
            # add ghost cells to the coarse meshes if the fine mesh has them
            if hasattr(meshes[0], 'edge_index_BC'):
                edge_BC_mid = meshes[0].node_xy[meshes[0].edge_index_BC].mean(1)
                meshes = interpolate_BC_location_multiscale(meshes, edge_BC_mid)
                meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
            else:
                raise ValueError("The meshes must have boundary conditions")            

        self.edge_BC = np.concatenate([mesh.edge_BC + self.edge_ptr[i] for i, mesh in enumerate(meshes)])
        self.face_BC = np.concatenate([mesh.face_BC + self.face_ptr[i] for i, mesh in enumerate(meshes)])
        self.edge_index_BC = self.edge_index[:,self.edge_type == 2].T

    def get_partitioning(self, meshes):
        """Get the partitioning of the meshes in the multiscale mesh
        
        Adds:
            node_ptr: index of first node of each mesh
            face_ptr: index of first face of each mesh
            edge_ptr: index of first edge of each mesh
        """
        self.node_ptr = np.cumsum([0] + [mesh.node_x.shape[0] for mesh in meshes])
        self.face_ptr = np.cumsum([0] + [mesh.face_x.shape[0] for mesh in meshes])
        self.edge_ptr = np.cumsum([0] + [mesh.edge_index.shape[1] for mesh in meshes])
        self.dual_edge_ptr = np.cumsum([0] + [mesh.dual_edge_index.shape[1] for mesh in meshes])
        
    def __repr__(self) -> str:
        return 'MultiscaleMesh object with {} meshes, {} nodes, {} edges, {} faces, and {} dual edges'.format(
            self.num_meshes, self.node_x.shape[0], self.edge_index.shape[1], self.face_x.shape[0], self.dual_edge_index.shape[1])
    
def rotate_mesh(mesh, angle):
    """Data augmentation: rotate the mesh by a given angle
    
    Args:
        mesh (Mesh): mesh object
        angle (float): angle in degrees
    """
    rotated_mesh = copy(mesh)

    angle = np.deg2rad(angle)

    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    rotated_mesh.node_x, rotated_mesh.node_y = np.dot(rot_matrix, np.array([mesh.node_x, mesh.node_y]))
    rotated_mesh.face_x, rotated_mesh.face_y = np.dot(rot_matrix, np.array([mesh.face_x, mesh.face_y]))
    
    rotated_mesh._get_derived_attributes()

    return rotated_mesh

def get_slopes(coords, DEM, neighborhood_size=200, min_neighbours=5):
    """
    Calculate the slope from points with x, y coordinates and z elevation (DEM).
    ------
    coords (np.array): x and y coordinates of each point
    DEM (np.array): elevation of each point
    neighborhood_size (float): maximum radius of the local neighborhood used for fitting the plane.
    min_neighbours (int): minimum neighborhood size
    """
    slope_x = []
    slope_y = []

    radius_graph = radius_neighbors_graph(coords, neighborhood_size, mode='connectivity', include_self=False)
    KNN = kneighbors_graph(coords, min_neighbours, mode='connectivity', include_self=False)

    for row in ((radius_graph.todense() + KNN.todense()) > 0):    
        A = np.column_stack((np.ones((row.sum(), 1)), coords[np.where(row)[1]]))
        b = DEM[np.where(row)[1]]
        coefficients, _, _, _ = lstsq(A, b)

        # The gradient of the plane is the coefficients for x and y
        dz_dx = coefficients[1]
        dz_dy = coefficients[2]

        slope_x.append(dz_dx)
        slope_y.append(dz_dy)

    return np.array(slope_x), np.array(slope_y)

def reorder_dict(dict):
    '''Change the key of a dictionary and sorts it by values order'''
    new_dict = {}
    
    #sort to exclude double values and order it
    dict = dict(sorted(dict.items()))

    #change keys from (x,y) format to i format
    for i, key in enumerate(dict.keys()):
        new_dict[i] = dict[key]
        
    return new_dict

def interpolate_variable(interpolated_points, points, value, method='nearest'):
    '''
    Interpolate variable at specific interpolated_points contained in points
    ------
    interpolated_points: np.array, shape (n, 2)
        points at which to interpolate data
    points: np.array, shape (m, 2)
        points at which the data is known
    variable: np.array, shape (m,)
        value of a variable for each point in the domain
    method: str
        choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)
    '''
    if isinstance(points, dict):
        points = get_coords(points)

    interpolated_variable = griddata(points, value, interpolated_points, method=method)
    
    # interpolate nan values
    mask = np.isnan(interpolated_variable)
    interpolated_variable[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), interpolated_variable[~mask])

    return interpolated_variable

def interpolate_temporal_variable(interpolated_points, points, temporal_value, method='nearest'):
    '''Interpolate temporal variable at specific interpolated_points contained in points
    ------
    interpolated_points: np.array, shape (n, 2)
        points at which to interpolate data
    points: np.array, shape (m, 2)
        points at which the data is known
    variable: np.array, shape (m, T)
        value of a variable for each point in the domain
    method: str
        choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)
    '''
    total_time = temporal_value.shape[1]

    interpolated_time_variable = np.stack([interpolate_variable(interpolated_points, points, temporal_value[:,time_step], method=method) for time_step in range(total_time)], 1)

    return interpolated_time_variable

def interpolate_mesh_attributes(fine_mesh, coarse_mesh, attribute, method='nearest'):
    """Interpolate the attribute from the fine mesh to the coarse mesh
    -------
    fine_mesh, coarse_mesh: Mesh
        fine and coarse meshes
    attribute: np.array (n,) or (n,T)
        attribute to interpolate
    method: str
        choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)

    Returns: interpolated_attribute
    """
    assert isinstance(fine_mesh, Mesh) and isinstance(coarse_mesh, Mesh), "Meshes must be of type Mesh"
    assert attribute.shape[0] == fine_mesh.num_faces, "Attribute must have the same number of nodes as the fine mesh"

    if attribute.ndim == 1:
        interpolated_attribute = interpolate_variable(coarse_mesh.face_xy, fine_mesh.face_xy, attribute, method=method)
    elif attribute.ndim == 2:
        interpolated_attribute = interpolate_temporal_variable(coarse_mesh.face_xy, fine_mesh.face_xy, attribute, method=method)
    else:
        raise ValueError("Attribute must be of shape (n,) or (n,T)")

    return interpolated_attribute

def interpolate_multiscale_attributes(meshes, *attributes, method='nearest'):
    """Interpolate and stack the attributes from the fine mesh to all coarse meshes in the list
    -------
    meshes: list of Mesh
        list of meshes from the coarse to finest
    attributes: list of np.array (n,) or (n,T)
        attributes to interpolate (must have the same number of nodes as the finest mesh)
    method: str
        choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)
    """
    assert isinstance(meshes, list), "meshes must be a list of meshes"
    assert len(meshes) > 0, "meshes must not be empty"
    assert all(isinstance(mesh, Mesh) for mesh in meshes), "meshes must be a list of meshes"

    if len(meshes) == 1:
        print("Only one mesh in the list. No multiscale interpolation needed.")
        return attributes
    
    fine_mesh = meshes[-1]

    interpolated_attributes = [np.concatenate([interpolate_mesh_attributes(fine_mesh, coarse_mesh, attribute, method=method) 
                                for coarse_mesh in meshes]) for attribute in attributes]
    
    return interpolated_attributes

def pool_multiscale_attributes(mesh, *attributes, reduce='mean'):
    """pool and stack the attributes from the fine mesh to all coarse meshes in the multiscale mesh
    -------
    mesh: MultiscaleMesh
        multiscale mesh containing the edges across meshes
    attributes: list of np.array (n,) or (n,T)
        attributes to pool
    reduce: str
        pooling type: choose from 'mean', 'max', 'add'
    """
    assert isinstance(mesh, MultiscaleMesh), "mesh must be a MultiscaleMesh"
    assert mesh.num_meshes > 1, "mesh must contain at least 2 meshes"
    
    coarse_to_fine = mesh.meshes[0].num_nodes < mesh.meshes[1].num_nodes
    attrs = []
    for attr in attributes:
        pooled_attributes = [torch.FloatTensor(attr)]
        for i in range(mesh.num_meshes-1):
            if coarse_to_fine:   # coarse to fine
                row, col = torch.LongTensor(mesh.intra_mesh_dual_edge_index)[:,mesh.intra_edge_ptr[-2-i]:mesh.intra_edge_ptr[-1-i]]
                pooled_attributes.append(scatter(src=pooled_attributes[-1][col-mesh.face_ptr[-2-i]], 
                                                index=row-mesh.face_ptr[-3-i], dim=0, 
                                                dim_size=mesh.meshes[-2-i].num_faces, reduce=reduce))
            else:   # fine to coarse
                row, col = torch.LongTensor(mesh.intra_mesh_dual_edge_index)[:,mesh.intra_edge_ptr[i]:mesh.intra_edge_ptr[i+1]]
                pooled_attributes.append(scatter(src=pooled_attributes[-1][col-mesh.face_ptr[i]], 
                                                index=row-mesh.face_ptr[i+1], dim=0, 
                                                dim_size=mesh.meshes[i+1].num_faces, reduce=reduce))

        pooled_attributes = torch.cat(pooled_attributes[::-1]) if coarse_to_fine else torch.cat(pooled_attributes)
        attrs.append(pooled_attributes)

    return attrs

def extract_single_scale_features_in_multimesh(mesh, scale, *features):
    """Extracts specified features at a single scale from a multiscale mesh.
    
    scale (int) : scale at which the features are extracted.
    features (torch.tensor): features to extract (e.g., WD, DEM, etc.)

    Example:
    WD, DEM = extract_single_scale_features_in_multimesh(mesh, scale, data.WD, data.DEM)
    """
    assert isinstance(mesh, MultiscaleMesh), "mesh must be a MultiscaleMesh object"
    assert scale < mesh.num_meshes, "scale must be smaller than the number of meshes in the multiscale mesh"

    scale = scale % mesh.num_meshes

    if len(features) == 1:
        features_at_scale = features[0][mesh.face_ptr[scale]:mesh.face_ptr[scale+1]]
    else:
        features_at_scale = [feature[mesh.face_ptr[scale]:mesh.face_ptr[scale+1]] for feature in features]

    return features_at_scale

def find_closest_nodes(all_points, reference_point, top_n=3):
    """Find the closest top_n nodes from all_points to the reference_point"""
    dist = np.sqrt(np.sum((all_points - reference_point)**2, axis=1))
    top = np.argsort(dist)[:top_n]
    return top

def interpolate_BC_location_multiscale(meshes, edge_BC_mid):
    """Find the location of the boundary condition edge for each multiscale mesh 
    by interpolation from the edge midpoints in the finest mesh
    
    Updates:
    edge_index_BC: np.array, shape (num_edges_BC, 2)
        index of the edges that have boundary conditions
    edge_BC: np.array, shape (num_edges_BC,)
        index of the edges that have boundary conditions
    edge_type: np.array, shape (num_edges_BC,)
        type of each edge (1:normal edges, 2:edge with boundary condition, 3:other boundary edges)
    face_BC: np.array, shape (num_faces_BC,)
        index of the faces that have boundary conditions
    """

    for mesh in copy(meshes):
        is_there_edge_BC = False
        # top_n is the number of closest nodes to consider for each edge midpoint
        top_n = 1

        while not is_there_edge_BC:
            possible_edges = np.array([find_closest_nodes(mesh.node_xy, edge, top_n=top_n) for edge in edge_BC_mid])
            mesh.edge_index_BC = np.array([[node for node in edge if (mesh.node_xy[node] == mesh.boundary_nodes).sum() == 2] for edge in possible_edges])
            mesh.edge_BC = np.array([np.where(((edge == mesh.edge_index.T).sum(1) == 2) | 
                                            ((edge[::-1] == mesh.edge_index.T).sum(1) == 2))[0] 
                                            for edge in mesh.edge_index_BC]).reshape(-1)
            if mesh.edge_BC.shape[0] == 1:
                is_there_edge_BC = True
            elif top_n > 10:
                raise ValueError('No edge found')
            else:
                top_n += 1
        
        mesh.edge_type[mesh.edge_BC] = 2
        mesh.face_BC = find_face_BC(mesh)

        # overwrite edge_BC_mid to ensure that the ghost cells don't derail in the coarse meshes
        edge_BC_mid = mesh.node_xy[mesh.edge_index_BC].mean(1)

    return meshes

def get_BC_edge_index(dual_edge_index, face_BC, undirected_BC=False):
    """
    Adds ghost cells to existing graph in correspondance of boundary condition (BC) faces

    Returns:
    updated dual_edge_index (with ghost cells)
    ghost_cells_ids: np.array of ghost cells ids

    ------
    dual_edge_index: np.array
        contains a list of edges of the dual graph
    face_BC: np.array
        contains a list of boundary faces (faces with boundary conditions)
    undirected_BC: bool
        if True, the information flow can go also to ghost nodes
    """
    num_faces = dual_edge_index.max() + 1
    dual_edge_index_BC = []
    ghost_cells_ids = []

    for i, face in enumerate(face_BC):
        dual_edge_index_BC.append([num_faces+i, face])
        ghost_cells_ids.append(num_faces+i)
        if undirected_BC:
            dual_edge_index_BC.append([face, num_faces+i])

    return np.array(dual_edge_index_BC).T, np.array(ghost_cells_ids)

def get_ghost_nodes(mesh):
    """Returns the ghost nodes ids"""
    num_BC_faces = len(mesh.face_BC)
    ghost_edge_index = []
    ghost_face_nodes = []

    ghost_nodes = mesh.nodes_per_face[mesh.face_BC]-2
    mesh.ghost_node_ids = [mesh.node_x.shape[0]-j-1 for j in range(ghost_nodes.sum())][::-1]

    for i in range(num_BC_faces):
        ghost_edge_index.append([mesh.ghost_node_ids[i], mesh.edge_index_BC[i,0]])

        # loop for polygons with more than 3 nodes
        for j in range(ghost_nodes[i]-1):
            ghost_edge_index.append([mesh.ghost_node_ids[i+j], mesh.ghost_node_ids[i+j+1]])

        ghost_edge_index.append([mesh.edge_index_BC[i,1], mesh.ghost_node_ids[i+ghost_nodes[:i+1].sum()-1]])
        ghost_face_nodes = mesh.ghost_node_ids[::-1] + mesh.edge_index_BC[i].tolist()

    return np.array(ghost_edge_index).T, np.array(ghost_face_nodes).reshape(-1)

def find_BC_other_nodes(mesh):
    """Returns the coordinates of the nodes that are in the boundary faces but not in the boundary edges"""
    assert mesh.face_BC is not [], "The boundary faces face_BC must be known"

    BC_edge_index = []
    the_other_node = [] # the nodes which is not in the BC edge

    for face in mesh.face_BC:
        face_nodes = get_face_nodes_mesh(mesh)[face,:mesh.nodes_per_face[face]].astype(int)

        # Add first element to simplify circular iterations
        face_nodes = np.concatenate((face_nodes, face_nodes[:1]))

        face_BC_dual_edge_index = mesh.dual_edge_index[:,np.where(mesh.dual_edge_index == face)[1]] 
        face_BC_neighbours = face_BC_dual_edge_index[face_BC_dual_edge_index != face]
        face_nodes_BC_neighbours = get_face_nodes_mesh(mesh)[face_BC_neighbours,:mesh.nodes_per_face[face]]

        for i in range(mesh.nodes_per_face[face]):
            is_BC_edge = not any([face_nodes[i] in neighbour and face_nodes[i+1] in neighbour \
                            for neighbour in face_nodes_BC_neighbours])
            
            # check that the edge is actually in the boundary edges
            edge = face_nodes[i:i+2]
            is_edge_in_BC = (((edge == mesh.edge_index_BC).sum(1) == 2) | ((edge[::-1] == mesh.edge_index_BC).sum(1) == 2)).any()
            
            if is_BC_edge and is_edge_in_BC:
                BC_edge_index.append(edge)
                # adding the unique prevents duplicates that can happen in the case of a the starting point being the other node itself
                the_other_node.append(np.unique(np.array([item for item in face_nodes if item not in edge]))) 

    # assert (mesh.edge_index_BC[:,::-1] == np.array(BC_edge_index)).all(), "The boundary condition edges are not the same with the ones from DHYDRO"

    return np.concatenate(the_other_node)

def find_face_BC(mesh):
    """Find the faces that have boundary conditions (face_BC), knowing the edges with boundary conditions (edge_index_BC)"""
    face_BC = []

    for i, face_nodes in enumerate(get_face_nodes_mesh(mesh)):
        face_nodes = face_nodes[~np.isnan(face_nodes)]
        
        # Add first element to simplify circular iterations
        face_nodes = np.concatenate((face_nodes, face_nodes[:1])).astype(int)

        # Find which face/s contains the boundary condition edge/s
        if np.array([(mesh.edge_index_BC == face_nodes[j:j+2]).sum(1)==2 or
                     (mesh.edge_index_BC[:,::-1] == face_nodes[j:j+2]).sum(1)==2
                     for j in range(len(face_nodes)-1)]).sum(0) == 1:
            face_BC.append(i)

    return np.array(face_BC)

def add_ghost_cells_mesh(mesh):
    """Adds ghost cells to the mesh by mirroring the boundary faces and nodes w.r.t. the boundary edges
    
    Updates:
    node_x, node_y: np.array, shape (num_nodes+num_ghost_nodes,)
    face_x, face_y: np.array, shape (num_faces+num_ghost_faces,)
    nodes_per_face: np.array, shape (num_faces+num_ghost_faces,)
    edge_index: np.array, shape (2, num_edges+num_ghost_edges)
    edge_type: np.array, shape (num_edges+num_ghost_edges,)
    dual_edge_index: np.array, shape (2, num_dual_edges+num_ghost_dual_edges)
        this is also converted to undirected
    face_nodes
    """
    if not mesh.added_ghost_cells:
        the_other_node = find_BC_other_nodes(mesh)

        face_BC_xy = mesh.face_xy[mesh.face_BC]
        node_BC_xy = mesh.node_xy[the_other_node]

        # faces are mirrored w.r.t. edge center
        face_symmetry_point = mesh.node_xy[mesh.edge_index_BC].mean(1)

        edge_outward_normal_faces = mesh.edge_outward_normal[mesh.edge_BC]

        # nodes are mirrored w.r.t. edge center (triangles) or edge nodes (quatrilaterals)
        node_symmetry_point = np.concatenate([item.mean(0) if mesh.nodes_per_face[mesh.face_BC][i] == 3
                                        else item for i, item in enumerate(mesh.node_xy[mesh.edge_index_BC])]).reshape(-1,2)

        edge_outward_normal_nodes = np.concatenate([np.repeat(item.reshape(1,-1), 2, axis=0) if mesh.nodes_per_face[mesh.face_BC][i] == 4
                                        else item for i, item in enumerate(mesh.edge_outward_normal[mesh.edge_BC])]).reshape(-1,2)

        distance_face_edge_BC = np.linalg.norm((face_BC_xy - face_symmetry_point), axis=1).reshape(-1,1)
        distance_node_edge_BC = np.linalg.norm((node_BC_xy - node_symmetry_point), axis=1).reshape(-1,1)

        normal_adapter = np.int32([1, 0])
        ghost_face_BC_xy = face_symmetry_point - edge_outward_normal_faces[:,normal_adapter]*distance_face_edge_BC
        ghost_node_BC_xy = node_symmetry_point - edge_outward_normal_nodes[:,normal_adapter]*distance_node_edge_BC

        distance_node_ghost_BC = np.linalg.norm((node_BC_xy - ghost_node_BC_xy), axis=1).reshape(-1,1)

        # The ghost nodes are not mirrored correctly so we flip the normal
        if (distance_node_ghost_BC < distance_node_edge_BC).any():
            edge_outward_normal_faces[:,(distance_node_ghost_BC < distance_node_edge_BC).squeeze()] *= -1
            edge_outward_normal_nodes[:,(distance_node_ghost_BC < distance_node_edge_BC).squeeze()] *= -1

            ghost_face_BC_xy = face_symmetry_point - edge_outward_normal_faces[:,normal_adapter]*distance_face_edge_BC
            ghost_node_BC_xy = node_symmetry_point - edge_outward_normal_nodes[:,normal_adapter]*distance_node_edge_BC

        mesh.node_x = np.concatenate((mesh.node_x, ghost_node_BC_xy[:,0]))
        mesh.node_y = np.concatenate((mesh.node_y, ghost_node_BC_xy[:,1]))

        mesh.face_x = np.concatenate((mesh.face_x, ghost_face_BC_xy[:,0]))
        mesh.face_y = np.concatenate((mesh.face_y, ghost_face_BC_xy[:,1]))

        mesh.nodes_per_face = np.concatenate((mesh.nodes_per_face, mesh.nodes_per_face[mesh.face_BC]))
        
        # update edge_index and dual_edge_index after adding ghost cells
        # dual_edge_index is converted to undirected
        mesh.dual_edge_index_BC, mesh.ghost_cells_ids = get_BC_edge_index(mesh.dual_edge_index, 
                                                                mesh.face_BC, undirected_BC=False)
        mesh.dual_edge_index = np.concatenate((mesh.dual_edge_index, mesh.dual_edge_index_BC), 1)
        ghost_edge_index, ghost_face_nodes = get_ghost_nodes(mesh)
        mesh.edge_index = np.concatenate((mesh.edge_index, ghost_edge_index), 1)
        mesh.edge_type = np.concatenate((mesh.edge_type, np.ones(ghost_edge_index.shape[1], dtype=np.int32)*4))        
        mesh.face_nodes = np.concatenate((mesh.face_nodes, ghost_face_nodes))
        
        mesh._get_derived_attributes()
        mesh.added_ghost_cells = True
    
    else:
        print("Ghost cells already added. Skipping...")

    return mesh

def remove_ghost_cells(mesh):
    """Remove all ghost cells from the mesh

    PERFORMS IN-PLACE MODIFICATION OF THE MESH
    
    Updates:
    node_x, node_y: np.array, shape (num_nodes,)
    face_x, face_y: np.array, shape (num_faces,)
    nodes_per_face: np.array, shape (num_faces,)
    edge_index: np.array, shape (2, num_edges)
    edge_type: np.array, shape (num_edges,)
    dual_edge_index: np.array, shape (2, num_dual_edges)
    face_nodes
    """
    if not mesh.added_ghost_cells:
        print("No ghost cells present in the mesh")
    else:
        num_ghost_cells = len(mesh.ghost_cells_ids)
        num_ghost_nodes = len(mesh.ghost_node_ids)
        num_face_nodes = mesh.nodes_per_face[-num_ghost_cells:].sum()

        mesh.node_x = mesh.node_x[:-num_ghost_nodes]
        mesh.node_y = mesh.node_y[:-num_ghost_nodes]

        mesh.face_x = mesh.face_x[:-num_ghost_cells]
        mesh.face_y = mesh.face_y[:-num_ghost_cells]

        mesh.nodes_per_face = mesh.nodes_per_face[:-num_ghost_cells]
        
        mesh.dual_edge_index = mesh.dual_edge_index[:,:-num_ghost_cells]
        
        mesh.edge_index = mesh.edge_index[:,:-(num_ghost_nodes+num_ghost_cells)]
        mesh.edge_type = mesh.edge_type[:-(num_ghost_nodes+num_ghost_cells)]

        mesh.face_nodes = mesh.face_nodes[:-num_face_nodes]
        
        mesh._get_derived_attributes()
        mesh.added_ghost_cells = False
    
    return mesh

def remove_ghost_cells_multiscale(mesh):
    """Remove all ghost cells from a Multiscale mesh"""
    assert isinstance(mesh, MultiscaleMesh), "Input mesh must be a MultiscaleMesh"
    
    new_meshes = [remove_ghost_cells(copy(m)) for m in mesh.meshes]

    new_mesh = MultiscaleMesh()
    new_mesh.stack_meshes(new_meshes)

    return new_mesh

def add_ghost_cells_attributes(mesh, *attributes):
    '''Corrects attribute value at ghost cells'''
    assert mesh.added_ghost_cells, "This function must be executed after add_ghost_cells_mesh"
    
    attribute_BC = [np.concatenate((attr, attr[mesh.face_BC]), axis=0) for attr in attributes]

    return attribute_BC

def update_ghost_cells_attributes(mesh, *attributes):
    '''Corrects attribute value at ghost cells'''
    assert mesh.added_ghost_cells, "This function must be executed after add_ghost_cells_mesh"
    
    for attr in attributes:
        attr[mesh.ghost_cells_ids] = attr[mesh.face_BC]

    return attributes

def convert_mesh_to_pyg(netcdf_file, DEM_file, BC, polygon_file=None, type_BC=2,
                        with_multiscale=False, number_of_multiscales=4,
                        neighborhood_size_slope=150, min_neighbours_slope=5):
    '''
    Creates a pytorch geometric Data object of a mesh simulation
    ------
    netcdf_file: str, path-like
        path to netcdf file
    DEM_file: str, path-like
        path to DEM file
    BC: np.array
        boundary condition array (e.g., hydrograph)
    polygon_file: str, path-like
        path to polygon file (only required if with_multiscale is True)
    type_BC: int
        type of boundary condition (1: water level, 2: discharge)
    with_multiscale: bool
        if True, data.mesh is a list of multiscale meshes
    number_of_multiscales: int
        number of multiscale meshes (default 4)
    neighborhood_size_slope: float
        radius around a point that determines which points account for local slope
    min_neighbours_slope: int
        minimum number of neighbours in slope evaluation
    '''
    data = Data()

    # Import mesh attributes (water depth, velocity, slopes, etc.)
    nc_dataset = xr.open_dataset(netcdf_file)

    WD = nc_dataset['mesh2d_waterdepth'].data.T
    VX = nc_dataset['mesh2d_ucx'].data.T
    VY = nc_dataset['mesh2d_ucy'].data.T

    mesh = Mesh()
    mesh._import_from_map_netcdf(netcdf_file)
    mesh._import_DEM(DEM_file)
    DEM = mesh.DEM
    mesh = add_ghost_cells_mesh(mesh)

    if with_multiscale:
        assert polygon_file is not None, 'polygon_file must be provided if with_multiscale is True'
        # create multiscale meshes
        meshes = create_mesh_dhydro(polygon_file, number_of_multiscales-1, for_simulation=False)
        meshes.append(copy(meshes[0]))
        meshes[-1]._import_from_map_netcdf(netcdf_file)
        meshes[-1].edge_outward_normal[meshes[-1].edge_BC] *= -1  # reverse the normal of the boundary edges
        meshes = meshes[::-1]

        # Add boundary conditions to multiscale meshes
        edge_BC_mid = mesh.node_xy[mesh.edge_index_BC].mean(1)
        meshes = interpolate_BC_location_multiscale(meshes, edge_BC_mid)
        meshes = [add_ghost_cells_mesh(mesh) for mesh in meshes]
        DEM, WD, VX, VY = add_ghost_cells_attributes(meshes[0], DEM, WD, VX, VY)

        # create multiscale mesh
        mesh = MultiscaleMesh()
        mesh.stack_meshes(meshes)

        data.node_ptr = torch.LongTensor(mesh.face_ptr)
        data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr)
        data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr)
        data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)
        
        # get multiscale attributes
        # mesh.DEM, WD, VX, VY = interpolate_multiscale_attributes(meshes, DEM, WD, VX, VY, method='nearest')
        mesh.DEM, WD, VX, VY = pool_multiscale_attributes(mesh, DEM, WD, VX, VY, reduce='mean')
        mesh.DEM = update_ghost_cells_attributes(mesh, mesh.DEM)[0] #correct ghost cells values after pooling
    else:
        mesh.DEM, WD, VX, VY = add_ghost_cells_attributes(mesh, mesh.DEM, WD, VX, VY)
    # slope_x, slope_y = get_slopes(mesh.face_xy, mesh.DEM, neighborhood_size=neighborhood_size_slope, 
    #                                   min_neighbours=min_neighbours_slope)
    
    data.DEM = torch.FloatTensor(mesh.DEM)
    data.WD = torch.FloatTensor(WD)
    data.VX = torch.FloatTensor(VX)
    data.VY = torch.FloatTensor(VY)
    # data.slopex = torch.FloatTensor(slope_x)
    # data.slopey = torch.FloatTensor(slope_y)
    
    # Assign other data properties
    data.edge_index = torch.LongTensor(mesh.dual_edge_index)
    data.face_distance = torch.FloatTensor(mesh.dual_edge_length)
    data.face_relative_distance = torch.FloatTensor(mesh.face_relative_distance)
    data.edge_slope = (data.DEM[data.edge_index][0] - data.DEM[data.edge_index][1])/data.face_distance
    # data.normal = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3])
    data.num_nodes = mesh.face_x.shape[0]
    data.area = torch.FloatTensor(mesh.face_area)

    data.mesh = mesh
    
    data.node_BC = torch.IntTensor(mesh.ghost_cells_ids)
    data.edge_BC_length = torch.FloatTensor(mesh.edge_length[mesh.edge_BC])
    if with_multiscale:
        data.node_BC = data.node_BC[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC only at the finest scale
        data.edge_BC_length = data.edge_BC_length[:len(mesh.ghost_cells_ids)//number_of_multiscales] # select BC+edge only at the finest scale
    data.BC = torch.FloatTensor(BC).unsqueeze(0).repeat(len(data.node_BC), 1, 1) # This repeats the same BC
    data.type_BC = torch.tensor(type_BC, dtype=torch.int)

    return data

def create_mesh_dataset(dataset_folder, n_sim, start_sim=1, 
                        with_multiscale=False, number_of_multiscales=4,
                        neighborhood_size_slope=150, min_neighbours_slope=9):
    '''
    Creates a list of pytorch geometric Data objects with n_sim simulations
    returns a mesh dataset
    ------
    dataset_folder: str, path-like
        path to raw dataset location
    n_sim: int
        number of simulations used in the dataset creation
    start_sim: int
        starting simulation id
    with_multiscale: bool
        if True, data.mesh is a list of multiscale meshes
    number_of_multiscales: int
        number of multiscale meshes (default 4)
    neighborhood_size_slope: float
        radius around a point that determines which points account for local slope
    min_neighbours_slope: int
        minimum number of neighbours in slope evaluation
    '''
    mesh_dataset = []

    for i in tqdm(range(start_sim,start_sim+n_sim)):
        netcdf_file = f'{dataset_folder}/Simulations/output_{i}_map.nc'
        DEM_file = f"{dataset_folder}\\DEM\\DEM_{i}.xyz"
        hydrograph_file = f"{dataset_folder}\\Hydrograph\\Hydrograph_{i}.txt"
        polygon_file = f"{dataset_folder}\\Geometry\\Polygon_{i}.pol"
        BC = np.loadtxt(hydrograph_file)
        BC[:,0] /= 60 # convert to minutes
        
        data = convert_mesh_to_pyg(netcdf_file, DEM_file, BC, polygon_file, type_BC=2,
                        with_multiscale=with_multiscale, number_of_multiscales=number_of_multiscales,
                        neighborhood_size_slope=neighborhood_size_slope, 
                        min_neighbours_slope=min_neighbours_slope)

        mesh_dataset.append(data)
    
    return mesh_dataset

def invert_scale_ordering(data):
    """Invert the ordering of the node and edge features in the multiscale mesh (from coarse to fine or viceversa).
    Use this function on the pyg_dataset obtained in create_datasets"""

    assert isinstance(data.mesh, MultiscaleMesh), "This function is valid only for MultiscaleMesh datasets."
    
    temp = Data()

    node_ptr = data.node_ptr
    edge_ptr = data.edge_ptr
    intra_edge_ptr = data.intra_edge_ptr

    temp.node_ptr = torch.LongTensor(np.cumsum([0]+[node_ptr[-i-1]-node_ptr[-i-2] for i in range(len(node_ptr)-1)]))
    temp.edge_ptr = torch.LongTensor(np.cumsum([0]+[edge_ptr[-i-1]-edge_ptr[-i-2] for i in range(len(edge_ptr)-1)]))
    temp.intra_edge_ptr = torch.LongTensor(np.cumsum([0]+[intra_edge_ptr[-i-1]-intra_edge_ptr[-i-2] for i in range(len(intra_edge_ptr)-1)]))

    temp.WD = torch.cat([data.WD[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.VX = torch.cat([data.VX[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.VY = torch.cat([data.VY[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.slopex = torch.cat([data.slopex[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.slopey = torch.cat([data.slopey[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.DEM = torch.cat([data.DEM[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])
    temp.area = torch.cat([data.area[node_ptr[i]:node_ptr[i+1]] for i in range(len(node_ptr)-1)][::-1])

    temp.BC = torch.flip(data.BC, [0])
    temp.node_BC = torch.stack([data.node_BC[i]-node_ptr[i+1]+temp.node_ptr[-i-1] for i in range(len(data.node_BC))])
    temp.type_BC = data.type_BC

    temp.edge_index = torch.cat([data.edge_index[:,edge_ptr[i]:edge_ptr[i+1]]-node_ptr[i]+temp.node_ptr[-i-2] for i in range(len(edge_ptr)-1)][::-1], 1)
    temp.face_distance = torch.cat([data.face_distance[edge_ptr[i]:edge_ptr[i+1]] for i in range(len(edge_ptr)-1)][::-1])
    temp.face_relative_distance = torch.cat([data.face_relative_distance[edge_ptr[i]:edge_ptr[i+1]] for i in range(len(edge_ptr)-1)][::-1])
    temp.edge_BC_length = torch.flip(data.edge_BC_length, [0])

    meshes = data.mesh.meshes[::-1]
    mesh = MultiscaleMesh()
    mesh.stack_meshes(meshes)
    temp.mesh = mesh
    temp.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)

    return temp

def create_dataset_folders(dataset_folder='datasets'):
    """Creates the folders for storing training and testing datasets"""
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


def save_database(dataset, name, out_path='datasets'):
    '''
    This function saves the geometric database into a pickle file
    The name of the file is given by the type of graph and number of simulations
    ------
    dataset: list
        list of geometric datasets for grid and mesh
    names: str
        name of saved dataset
    out_path: str, path-like
        output file location
    '''
    n_sim = len(dataset)
    path = f"{out_path}/{name}.pkl"
    
    if os.path.exists(path):
        os.remove(path)
    elif not os.path.exists(out_path):
        os.mkdir(out_path)
    
    pickle.dump(dataset, open(path, "wb" ))
        
    return None