import os
import random
from dataclasses import dataclass, field
import smplx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from pysdf import SDF
import scipy.sparse
import cubvh
import pymeshlab as ml
import copy
import trimesh
from pytorch3d.ops import knn_points
import pickle
from threestudio.utils.ops import chunk_batch, scale_tensor
from threestudio.models.isosurface import (
    IsosurfaceHelper,
    MarchingCubeCPUHelper,
    MarchingTetrahedraHelper,
    MarchingTetrahedraShellHelper,
    GshellHelper
)
import pymeshfix
import collections
from scipy.spatial import cKDTree

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import alphashape

def fit_plane_and_create_elliptical_cylinder(points, thickness_scale=1.02, margin=0.1, sections=64, round_flag=True):
    center = np.mean(points, axis=0)

    pca = PCA(n_components=3)
    pca.fit(points - center)
    u = pca.components_[0]  
    v = pca.components_[1]  
    n = pca.components_[2]  
    
    proj_uv = np.column_stack([(points - center) @ u, (points - center) @ v])
    if round_flag:
        r1 = np.sqrt(max(np.sum(proj_uv ** 2, axis=1))) * (1 + margin)
        r2 = np.sqrt(max(np.sum(proj_uv ** 2, axis=1))) * (1 + margin)
    else:
        min_xy = np.min(proj_uv, axis=0)
        max_xy = np.max(proj_uv, axis=0)
        diff_xy = max_xy - min_xy
        min_xy = min_xy - margin * diff_xy
        max_xy = max_xy + margin * diff_xy
        r1 = (max_xy[0] - min_xy[0]) / 2.0
        r2 = (max_xy[1] - min_xy[1]) / 2.0

    proj_n = (points - center) @ n
    t_min, t_max = np.min(proj_n), np.max(proj_n)
    thickness = t_max - t_min

    cylinder = trimesh.creation.cylinder(radius=1.0, height=thickness * thickness_scale, sections=sections)
    
    T_scale = np.eye(4)
    T_scale[0, 0] = r1
    T_scale[1, 1] = r2
    
    T_rot = np.eye(4)
    T_rot[:3, 0] = u
    T_rot[:3, 1] = v
    T_rot[:3, 2] = n

    T_trans = np.eye(4)
    T_trans[:3, 3] = center

    T_total = T_trans @ T_rot @ T_scale

    # 10. 应用变换到单位圆柱
    elliptical_cylinder = cylinder.copy()
    elliptical_cylinder.apply_transform(T_total)
    
    return elliptical_cylinder, n, thickness

def fit_plane_and_create_custom_cylinder(points, alpha_param=None, margin=0.0, thickness_scale=1.02, sections=64):
    center = np.mean(points, axis=0)
    
    pca = PCA(n_components=3)
    pca.fit(points - center)
    u = pca.components_[0]
    v = pca.components_[1]
    n = pca.components_[2]
    
    uv = np.column_stack(((points - center) @ u, (points - center) @ v))
    uv = uv * (1 + margin)

    proj_n = (points - center) @ n
    t_min = np.min(proj_n)
    t_max = np.max(proj_n)
    thickness = t_max - t_min
    thickness = thickness * thickness_scale
    t_offset = (t_max + t_min) / 2.0 

    uv_list = [tuple(p) for p in uv]
    if alpha_param is None:
        alpha_param = alphashape.optimizealpha(uv_list)
    hull = alphashape.alphashape(uv_list, alpha_param)
    if hull.geom_type == 'MultiPolygon':
        hull = max(hull, key=lambda p: p.area)
    polygon_2d = hull

    mesh_2d_extruded = trimesh.creation.extrude_polygon(polygon_2d, height=thickness)
    T_center = np.eye(4)
    T_center[2, 3] = -thickness / 2.0
    mesh_2d_extruded.apply_transform(T_center)

    T_rot = np.eye(4)
    T_rot[:3, 0] = u
    T_rot[:3, 1] = v
    T_rot[:3, 2] = n

    T_trans = np.eye(4)
    T_trans[:3, 3] = center + t_offset * n
    T_total = T_trans @ T_rot
    mesh_2d_extruded.apply_transform(T_total)
    
    return mesh_2d_extruded

def save_pc(pc, save_path, color=None, line_flag=False):
    obj_str = ""
    for i in range(pc.shape[0]):
        if color is None:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]}"
        else:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]} {color[i][0]} {color[i][1]} {color[i][2]}"
        obj_str += "\n"
    obj_str += "\n"
    if line_flag:
        for i in range(1, pc.shape[0] + 1):
            obj_str += f"l {i} {i+1}"
            obj_str += "\n"
        obj_str += "\n"
    with open(save_path, "w") as f:
        f.write(obj_str)
    return 

def smooth_mesh(mesh, mask=None, iterations=10, lambda_param=0.5):
    vertices = mesh.vertices
    faces = mesh.faces

    for _ in range(iterations):
        vertex_neighbors = {i: set() for i in range(len(vertices))}
        for face in faces:
            for i in range(3):
                vertex_neighbors[face[i]].add(face[(i+1) % 3])
                vertex_neighbors[face[i]].add(face[(i+2) % 3])

        new_vertices = np.copy(vertices)
        if mask is None:
            for i, neighbors in vertex_neighbors.items():
                if len(neighbors) > 0:
                    new_vertices[i] = (1 - lambda_param) * vertices[i] + lambda_param * np.mean(vertices[list(neighbors)], axis=0)
        else:
            for i, neighbors in vertex_neighbors.items():
                if len(neighbors) > 0 and mask[i] == True:
                    new_vertices[i] = (1 - lambda_param) * vertices[i] + lambda_param * np.mean(vertices[list(neighbors)], axis=0)

        vertices = new_vertices

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def find_open_holes(mesh):
    faces = mesh.faces
    edge_counts = collections.defaultdict(int)
    for face in faces:
        v0, v1, v2 = face
        edges = [(v0, v1), (v1, v2), (v2, v0)]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))
            edge_counts[sorted_edge] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

    adj = collections.defaultdict(list)
    for edge in boundary_edges:
        v0, v1 = edge
        adj[v0].append(v1)
        adj[v1].append(v0)
    
    holes = []
    while adj:
        start_vertex = next(iter(adj))
        current_vertex = start_vertex
        hole = [current_vertex]
        while True:
            next_vertex = adj[current_vertex][0]
            hole.append(next_vertex)
            adj[current_vertex].remove(next_vertex)
            if not adj[current_vertex]:
                del adj[current_vertex]
            adj[next_vertex].remove(current_vertex)
            if not adj[next_vertex]:
                del adj[next_vertex]
            current_vertex = next_vertex
            if current_vertex == start_vertex:
                break
        if hole[0] == hole[-1]:
            hole = hole[:-1]
        holes.append(hole)
    return holes

@threestudio.register("implicit-gshell-cloth")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

        # improve the resolution of DMTET at these steps
        progressive_resolution: dict = field(default_factory=dict)
        # progressive_resolution_steps: Optional[Any] = None

        # new
        test_save_path: str = "./.threestudio_cache"
        smplx_path: str = "./load/smplx_models"
        gender: str = "neutral"
        start_temp_loss_step: int = 0
        update_temp_loss_step: int = 1000
        msdf_type: str = "explicit"
        init_msdf_type: str = "mesh"
        isosurface_method: str = "gshell"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.sdf_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )

        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )

        self.finite_difference_normal_eps: Optional[float] = None
        self.cached_sdf = None

        self.smplx_model = smplx.create(
                self.cfg.smplx_path,
                model_type="smplx",
                gender=self.cfg.gender,
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,  # explicitly control hand pose
                flat_hand_mean=True,  # use a flatten hand default pose
            )
        with open(f'{self.cfg.smplx_path}/smplx/smplx_watertight.pkl', 'rb') as f:
            tmp_data = pickle.load(f, encoding='latin1')
        self.smplx_model.faces = tmp_data["smpl_F"].numpy()
        
        self.temp_mesh = None
        self.temp_mesh_sdf = None

        self.temp_cyl_mesh = None
        self.temp_cyl_mesh_sdf = None

        self.register_buffer("scale", torch.ones(3).float())
        self.register_buffer("trans", torch.zeros(3).float())

        if self.cfg.msdf_type == "implicit":
            self.msdf_encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.msdf_network = get_mlp(
                self.msdf_encoding.n_output_dims, 1, self.cfg.mlp_network_config
            )
        elif self.cfg.msdf_type == "explicit":
            self.register_parameter(
                "msdf",
                nn.Parameter(
                    torch.zeros(
                        (2000000, 1),
                        dtype=torch.float32,
                    )
                ),
            )

    def openmesh2clothmesh(self, init_mesh, scale_factor1=1, scale_factor2=1, hollow_flag=True, smooth_flag=True, comp_num=1):
        import trimesh
        os.makedirs(f"{self.cfg.test_save_path}/obj/init_mesh", exist_ok=True)
        init_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh.obj")
        def close_hole(mesh_path, iter=10):
            tmp_ms = ml.MeshSet()
            for i in range(iter):
                tmp_ms.clear()
                tmp_ms.load_new_mesh(mesh_path)
                tmp_ms.apply_filter('compute_selection_from_mesh_border')
                v_selection = tmp_ms.current_mesh().vertex_selection_array()
                tmp_ms.apply_filter('meshing_close_holes', maxholesize = v_selection.sum(), selected = True)
                close_mesh = trimesh.Trimesh(vertices=tmp_ms.current_mesh().vertex_matrix(), faces=tmp_ms.current_mesh().face_matrix())
                close_mesh.fill_holes()
                close_mesh.export(mesh_path)
                tmp_ms.clear()
                tmp_ms.load_new_mesh(mesh_path)
                tmp_ms.apply_filter('compute_selection_from_mesh_border')
                v_selection = tmp_ms.current_mesh().vertex_selection_array()
                if close_mesh.is_watertight and v_selection.sum() == 0:
                    break
                else:
                    print(f"close_hole_iter: {i}, boundary_vert: {v_selection.sum()}")
        ms = ml.MeshSet()
        if hollow_flag:
            init_mesh_scale1 = copy.copy(init_mesh)
            init_mesh_scale1.vertices = init_mesh_scale1.vertices + init_mesh_scale1.vertex_normals * (scale_factor1 - 1)
            init_mesh_scale1.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale1.obj")
            init_mesh_scale2 = copy.copy(init_mesh)
            init_mesh_scale2.vertices = init_mesh_scale2.vertices + init_mesh_scale2.vertex_normals * (scale_factor2 - 1)
            init_mesh_scale2.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale2.obj")
            ms.clear()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh.obj")
            ms.apply_filter('compute_selection_from_mesh_border')
            v_selection = ms.current_mesh().vertex_selection_array()
            border_vertices_idx = list(np.unique(np.where(v_selection==True)[0]))
            replace_dict = {}
            for i in border_vertices_idx:
                replace_dict[i + init_mesh.vertices.shape[0]] = i 
            tmp_faces = init_mesh_scale1.faces + init_mesh.vertices.shape[0]
            for k, v in replace_dict.items():
                tmp_faces = np.where(tmp_faces==k, v, tmp_faces)
            obj_str = ""
            for i in range(init_mesh.vertices.shape[0]):
                if i in border_vertices_idx:
                    tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                    obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                    obj_str += "\n"
                else:
                    obj_str += f"v {init_mesh_scale1.vertices[i][0]} {init_mesh_scale1.vertices[i][1]} {init_mesh_scale1.vertices[i][2]}"
                    obj_str += "\n"
            for i in range(init_mesh.vertices.shape[0]):
                if i in border_vertices_idx:
                    tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                    obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                    obj_str += "\n"
                else:
                    obj_str += f"v {init_mesh_scale2.vertices[i][0]} {init_mesh_scale2.vertices[i][1]} {init_mesh_scale2.vertices[i][2]}"
                    obj_str += "\n"
            for i in range(init_mesh_scale1.faces.shape[0]):
                obj_str += f"f {init_mesh_scale1.faces[i][0]+1} {init_mesh_scale1.faces[i][2]+1} {init_mesh_scale1.faces[i][1]+1}"
                obj_str += "\n"
            for i in range(tmp_faces.shape[0]):
                obj_str += f"f {tmp_faces[i][0]+1} {tmp_faces[i][1]+1} {tmp_faces[i][2]+1}"
                obj_str += "\n"
            obj_str += "\n"
            with open(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj", "w") as f:
                f.write(obj_str)
            init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
            # init_close_mesh = pymeshfix.MeshFix(init_close_mesh.vertices, init_close_mesh.faces)
            # init_close_mesh.repair(verbose=True)
            # init_close_mesh = trimesh.Trimesh(vertices=init_close_mesh.v, faces=init_close_mesh.f)
            init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
        else:
            init_mesh_scale = copy.copy(init_mesh)
            init_mesh_scale.vertices = init_mesh_scale.vertices + init_mesh_scale.vertex_normals * (scale_factor1 - 1)
            init_mesh_scale.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            
            if comp_num > 1:
                components = init_mesh_scale.split(only_watertight=False)
                largest_components = heapq.nlargest(comp_num, components, key=lambda comp: len(comp.faces))
                tmp_list = []
                for idx in range(comp_num):
                    tmp_init_close_mesh = pymeshfix.MeshFix(largest_components[idx].vertices, largest_components[idx].faces)
                    tmp_init_close_mesh.repair(verbose=True)
                    tmp_list.append(trimesh.Trimesh(vertices=tmp_init_close_mesh.v, faces=tmp_init_close_mesh.f))
                init_close_mesh = trimesh.util.concatenate(tmp_list)
                init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
            else:
                init_close_mesh = pymeshfix.MeshFix(init_mesh_scale.vertices, init_mesh_scale.faces)
                init_close_mesh.repair(verbose=True)
                init_close_mesh = trimesh.Trimesh(vertices=init_close_mesh.v, faces=init_close_mesh.f)
                init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")

        return init_close_mesh


    def initialize_shape(self, verts=None, faces=None, system_cfg=None, opt_smplx_mesh=None) -> None:
        sdf = None
        import trimesh
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert system_cfg is not None
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)
            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert system_cfg is not None
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params
            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert system_cfg is not None
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        elif self.cfg.shape_init.startswith("human:"):
            assert verts is not None 
            assert faces is not None
            assert system_cfg is not None

            if self.cfg.shape_init.startswith("human:bbox:"):
                shape_init_strlist = self.cfg.shape_init.split(":")
                comp_num = int(shape_init_strlist[2])
                bbox_info = system_cfg.cloth.bbox_info
                bbox_extents = np.array([bbox_info[0] * 2, bbox_info[1] * 2, bbox_info[2] * 2])
                bbox_transform = np.array([bbox_info[3], bbox_info[4], bbox_info[5]])
                bbox_max = np.array([bbox_info[0], bbox_info[1], bbox_info[2]]) + bbox_transform
                bbox_min = np.array([-bbox_info[0], -bbox_info[1], -bbox_info[2]]) + bbox_transform
                human_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                inside = np.all((verts >= bbox_min) & (verts <= bbox_max), axis=1)
                inside_faces = [f for f in faces if np.all(inside[f])]
                used_vertices = np.unique(np.hstack(inside_faces))
                new_vertices = verts[used_vertices]
                vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
                new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
                components = new_mesh.split(only_watertight=False)
                largest_components = heapq.nlargest(comp_num, components, key=lambda comp: len(comp.faces))
                largest_component = trimesh.util.concatenate(largest_components)
                print(f"Inside largest component verts num: {largest_component.vertices.shape[0]}", f"Inside verts num: {np.where(inside==True)[0].shape[0]}", f"Outside verts num: {np.where(inside==False)[0].shape[0]}", f"Total verts num: {verts.shape[0]}")
                expand_scale = shape_init_strlist[3].split("_")
                if len(expand_scale) == 1:
                    expand_scale = float(expand_scale[0])
                    largest_component = self.openmesh2clothmesh(largest_component, expand_scale, None, hollow_flag=False)
                elif len(expand_scale) == 2:
                    expand_scale1 = float(expand_scale[0])
                    expand_scale2 = float(expand_scale[1])
                    largest_component = self.openmesh2clothmesh(largest_component, expand_scale1, expand_scale2, hollow_flag=True)
                centroid = (largest_component.vertices.max(0) + largest_component.vertices.min(0)) / 2
                largest_component.vertices = largest_component.vertices - centroid
                scale = np.abs(largest_component.vertices).max()
                largest_component.vertices = largest_component.vertices / scale * self.cfg.shape_init_params
                self.scale = torch.ones(3).float().to(self.device) * (1 / self.cfg.shape_init_params * scale)
                self.trans = torch.from_numpy(centroid).float().to(self.device)

                sdf = SDF(largest_component.vertices, largest_component.faces)
                self.temp_mesh = largest_component

                def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                    # add a negative signed here
                    # as in pysdf the inside of the shape has positive signed distance
                    return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                        points_rand
                    )[..., None]

                get_gt_sdf = func
            elif self.cfg.shape_init.startswith("human:smplx:"):
                self.smplx_model = smplx.create(
                    self.cfg.smplx_path,
                    model_type="smplx",
                    gender=self.cfg.gender,
                    use_face_contour=False,
                    num_betas=10,
                    num_expression_coeffs=10,
                    ext="npz",
                    use_pca=False,  # explicitly control hand pose
                    flat_hand_mean=True,  # use a flatten hand default pose
                )
                assert isinstance(self.cfg.shape_init_params, float)
                shape_init_strlist = self.cfg.shape_init.split(":")
                body_pose = np.zeros((21, 3), dtype=np.float32)
                if shape_init_strlist[2] == "apose":
                    body_pose[0, 1] = 0.2
                    body_pose[0, 2] = 0.1
                    body_pose[1, 1] = -0.2
                    body_pose[1, 2] = -0.1
                    body_pose[15, 2] = -1
                    body_pose[16, 2] = 1
                    # body_pose[15, 2] = -0.7853982
                    # body_pose[16, 2] = 0.7853982
                    # body_pose[19, 0] = 1.0
                    # body_pose[20, 0] = 1.0
                elif shape_init_strlist[2] == 'tpose':
                    pass
                elif shape_init_strlist[2] == 'dapose':
                    body_pose[0, 1] = 0.2
                    body_pose[0, 2] = 0.2
                    body_pose[1, 1] = -0.2
                    body_pose[1, 2] = -0.2
                    body_pose[15, 2] = -0.4
                    body_pose[16, 2] = 0.4
                elif shape_init_strlist[2] == 'dapose1':
                    body_pose[0, 1] = 0.2
                    body_pose[0, 2] = 0.2
                    body_pose[1, 1] = -0.2
                    body_pose[1, 2] = -0.2
                    body_pose[15, 2] = -0.7
                    body_pose[16, 2] = 0.7
                elif shape_init_strlist[2] == 'dapose2':
                    body_pose[0, 1] = 0.2
                    body_pose[0, 2] = 0.2
                    body_pose[1, 1] = -0.2
                    body_pose[1, 2] = -0.2
                    body_pose[15, 2] = -0.8
                    body_pose[16, 2] = 0.8

                jaw_pose = torch.tensor([0, 0, 0], dtype=torch.float32).unsqueeze(0).to(self.smplx_model.shapedirs.device)
                smplx_output = self.smplx_model(body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0).to(self.smplx_model.shapedirs.device), jaw_pose=jaw_pose, return_verts=True)
                # smplx_output = self.smplx_model(body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0), return_verts=True)
                smplx_vertices = smplx_output.vertices.detach().cpu().numpy()[0]
                smplx_faces = self.smplx_model.faces

                smplx_mesh = trimesh.Trimesh(
                    vertices=smplx_vertices,
                    faces=smplx_faces,
                )
                centroid = smplx_mesh.vertices.mean(0)
                smplx_mesh.vertices = smplx_mesh.vertices - centroid

                # adjust the position of mesh
                if shape_init_strlist[2] == "apose":
                    smplx_mesh.vertices[:,1] = smplx_mesh.vertices[:,1] + 0.3
                elif shape_init_strlist[2] == "dapose":
                    smplx_mesh.vertices[:,1] = smplx_mesh.vertices[:,1] + 0.4
                else:
                    smplx_mesh.vertices[:,1] = smplx_mesh.vertices[:,1] + 0.4

                # align to up-z and front-x
                dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
                dir2vec = {
                    "+x": np.array([1, 0, 0]),
                    "+y": np.array([0, 1, 0]),
                    "+z": np.array([0, 0, 1]),
                    "-x": np.array([-1, 0, 0]),
                    "-y": np.array([0, -1, 0]),
                    "-z": np.array([0, 0, -1]),
                }
                z_, x_ = (
                    dir2vec["+y"],
                    dir2vec["+z"],
                )
                y_ = np.cross(z_, x_) 
                std2mesh = np.stack([x_, y_, z_], axis=0).T
                mesh2std = np.linalg.inv(std2mesh)

                # scaling
                scale = np.abs(smplx_mesh.vertices).max()
                smplx_mesh.vertices = smplx_mesh.vertices / scale * self.cfg.shape_init_params
                smplx_mesh.vertices = np.dot(mesh2std, smplx_mesh.vertices.T).T

                human_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                human_verts_tensor = torch.from_numpy(verts).to(self.device).float()
                smplx_verts_tensor = torch.from_numpy(smplx_mesh.vertices).to(self.device).float()
                knn = knn_points(human_verts_tensor[None], smplx_verts_tensor[None], K=1)
                with open(os.path.join(self.cfg.smplx_path, "smplx/smplx_cloth_mask.pkl"), 'rb') as f:
                    cloth_dict = pickle.load(f)
                cloth_mask = torch.zeros(human_verts_tensor.shape[0]).to(self.device)
                cloth_list = shape_init_strlist[3].split("+")
                for cloth in cloth_list:
                    if "skirt" in cloth:
                        cloth_mask = torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["skirt"]).to(self.device) == True)[0]))
                        cloth_mask = torch.logical_and(human_verts_tensor[:,2] < human_verts_tensor[cloth_mask].max(0)[0][2], human_verts_tensor[:,2] > human_verts_tensor[cloth_mask].min(0)[0][2])
                        if cloth == "super_long_skirt":
                            hem_height = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["long_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2] - 0.3
                        elif cloth == "long_skirt":
                            hem_height1 = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["long_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2]
                            hem_height2 = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["short_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2]
                            hem_height = hem_height1 * 0.5 + hem_height2 * 0.5
                        elif cloth == "short_skirt":
                            hem_height1 = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["short_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2]
                            hem_height2 = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["super_short_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2]
                            hem_height = hem_height1 * 0.5 + hem_height2 * 0.5
                        elif cloth == "super_short_skirt":
                            hem_height = verts[torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["super_short_bottom"]).to(self.device) == True)[0])).cpu().numpy()].min(0)[2]
                    else:       
                        cloth_mask = torch.logical_or(cloth_mask, torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict[cloth]).to(self.device) == True)[0]))
                inside_faces = [f for f in faces if np.all(cloth_mask.cpu().numpy()[f])]
                used_vertices = np.unique(np.hstack(inside_faces))
                new_vertices = verts[used_vertices]
                vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
                new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
                components = new_mesh.split(only_watertight=False)
                comp_num = 1
                if "gloves" in cloth_list:
                    comp_num += 1
                if "shoes" in cloth_list:
                    comp_num += 1
                if "long_shoes" in cloth_list:
                    comp_num += 1
                if "super_long_shoes" in cloth_list:
                    comp_num += 1
                largest_components = heapq.nlargest(comp_num, components, key=lambda comp: len(comp.faces))
                new_mesh = trimesh.util.concatenate(largest_components)
                if 'hem_height' in locals():
                    def deform_skirt(mesh, hem_height, expand_scale):
                        tmp_vertices = np.copy(mesh.vertices)
                        top, bottom = tmp_vertices[:, 2].max(), tmp_vertices[:, 2].min()
                        center = tmp_vertices.mean(0)
                        tmp_vertices[:, 2] = tmp_vertices[:, 2] + (((top - tmp_vertices[:, 2]) / (top - bottom)) ** 1) * (hem_height - bottom)
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] - center[:2]
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] * (((top - tmp_vertices[:, 2]) / (top - bottom))[:, None] * (expand_scale - 1) + 1)
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] + center[:2]
                        temp_mesh = trimesh.Trimesh(vertices=tmp_vertices, faces=mesh.faces)
                        return temp_mesh
                    new_mesh = deform_skirt(new_mesh, hem_height, expand_scale=1.2)
                expand_scale = shape_init_strlist[4].split("_")
                if len(expand_scale) == 1:
                    expand_scale = float(expand_scale[0])
                    new_mesh = self.openmesh2clothmesh(new_mesh, expand_scale, None, hollow_flag=False)
                elif len(expand_scale) == 2:
                    expand_scale1 = float(expand_scale[0])
                    expand_scale2 = float(expand_scale[1])
                    new_mesh = self.openmesh2clothmesh(new_mesh, expand_scale1, expand_scale2, hollow_flag=True)
                centroid = (new_mesh.vertices.max(0) + new_mesh.vertices.min(0)) / 2
                new_mesh.vertices = new_mesh.vertices - centroid
                scale = np.abs(new_mesh.vertices).max()
                new_mesh.vertices = new_mesh.vertices / scale * self.cfg.shape_init_params
                self.scale = torch.ones(3).float().to(self.device) * (1 / self.cfg.shape_init_params * scale)
                self.trans = torch.from_numpy(centroid).float().to(self.device)
                sdf = SDF(new_mesh.vertices, new_mesh.faces)
                self.temp_mesh = new_mesh

                def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                    # add a negative signed here
                    # as in pysdf the inside of the shape has positive signed distance
                    return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                        points_rand
                    )[..., None]
                get_gt_sdf = func
            elif self.cfg.shape_init.startswith("human:opt_smplx:"):
                assert opt_smplx_mesh is not None
                assert isinstance(self.cfg.shape_init_params, float)
                shape_init_strlist = self.cfg.shape_init.split(":")
                smplx_mesh = trimesh.Trimesh(
                        vertices=opt_smplx_mesh.v_pos.detach().cpu().numpy(),
                        faces=opt_smplx_mesh.t_pos_idx.detach().cpu().numpy(),
                    )
                self.smplx_mesh = smplx_mesh
                human_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                human_verts_tensor = torch.from_numpy(verts).to(self.device).float()
                smplx_verts_tensor = torch.from_numpy(smplx_mesh.vertices).to(self.device).float()
                knn = knn_points(human_verts_tensor[None], smplx_verts_tensor[None], K=1)
                with open(os.path.join(self.cfg.smplx_path, "smplx/smplx_cloth_mask.pkl"), 'rb') as f:
                    cloth_dict = pickle.load(f)
                cloth_mask = torch.zeros(human_verts_tensor.shape[0]).to(self.device)
                cloth_list = shape_init_strlist[3].split("+")
                zero_joint = torch.einsum('ik,ji->jk', [opt_smplx_mesh.v_pos, self.smplx_model.J_regressor])[0]
                for cloth in cloth_list:
                    tmp_mask = torch.zeros(human_verts_tensor.shape[0]).to(self.device)
                    if "skirt" in cloth:
                        tmp_mask = torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["skirt"]).to(self.device) == True)[0])
                        # tmp_mask = torch.logical_and(human_verts_tensor[:,2] < human_verts_tensor[tmp_mask].max(0)[0][2], human_verts_tensor[:,2] > human_verts_tensor[tmp_mask].min(0)[0][2])
                        if cloth == "super_long_skirt":
                            hem_height = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["long_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2] - 0.3
                            skirt_expand_scale = 1.2               
                        elif cloth == "long_skirt":
                            hem_height1 = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["long_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2]
                            hem_height2 = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["short_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2]
                            hem_height = hem_height1 * 0.5 + hem_height2 * 0.5
                            skirt_expand_scale = 1.2
                        elif cloth == "short_skirt":
                            hem_height1 = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["short_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2]
                            hem_height2 = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["super_short_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2]
                            hem_height = hem_height1 * 0.5 + hem_height2 * 0.5
                            skirt_expand_scale = 1.2
                        elif cloth == "super_short_skirt":
                            hem_height = verts[torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict["super_short_bottom"]).to(self.device) == True)[0]).cpu().numpy()].min(0)[2]
                            skirt_expand_scale = 1.2
                    else:       
                        tmp_mask = torch.isin(knn.idx[0, :, 0], torch.where(torch.from_numpy(cloth_dict[cloth]).to(self.device) == True)[0])

                    if "skirt" in cloth or "bottom" in cloth:
                        tmp_mask = torch.logical_and(tmp_mask, human_verts_tensor[:,2] <= (zero_joint[2] - 0.02))
                    elif "top" in cloth:
                        tmp_mask = torch.logical_and(tmp_mask, human_verts_tensor[:,2] > (zero_joint[2] - 0.05))

                    cloth_mask = torch.logical_or(cloth_mask, tmp_mask)
                inside_faces = [f for f in faces if np.all(cloth_mask.cpu().numpy()[f])]
                used_vertices = np.unique(np.hstack(inside_faces))
                new_vertices = verts[used_vertices]
                vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
                new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
                components = new_mesh.split(only_watertight=False)
                comp_num = 1
                if "gloves" in cloth_list:
                    comp_num += 1
                if "shoes" in cloth_list:
                    comp_num += 1
                if "long_shoes" in cloth_list:
                    comp_num += 1
                if "super_long_shoes" in cloth_list:
                    comp_num += 1
                largest_components = heapq.nlargest(comp_num, components, key=lambda comp: len(comp.faces))
                new_mesh = trimesh.util.concatenate(largest_components)
                if "skirt" in shape_init_strlist[3]:
                    def deform_skirt(mesh, hem_height, skirt_expand_scale):
                        tmp_mask = (mesh.vertices[:, 2] <= (zero_joint[2].detach().cpu().numpy() - 0.02))
                        tmp_vertices = np.copy(mesh.vertices[tmp_mask])
                        top, bottom = tmp_vertices[:, 2].max(), tmp_vertices[:, 2].min()
                        tmp_vertices[:, 2] = tmp_vertices[:, 2] + ((top - tmp_vertices[:, 2]) / (top - bottom)) * (hem_height - bottom)
                        center = tmp_vertices.mean(0)
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] - center[:2]
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] * (((top - tmp_vertices[:, 2]) / (top - bottom))[:, None] * (skirt_expand_scale - 1) + 1)
                        tmp_vertices[:, :2] = tmp_vertices[:, :2] + center[:2]
                        tmp = np.copy(mesh.vertices)
                        tmp[tmp_mask] = tmp_vertices
                        temp_mesh = trimesh.Trimesh(vertices=tmp, faces=mesh.faces)
                        return temp_mesh
                    new_mesh = deform_skirt(new_mesh, hem_height, skirt_expand_scale)
                expand_scale = shape_init_strlist[4].split("_")
                if len(expand_scale) == 1:
                    expand_scale = float(expand_scale[0])
                    new_mesh = self.openmesh2clothmesh(new_mesh, expand_scale, None, hollow_flag=False, comp_num=comp_num)
                elif len(expand_scale) == 2:
                    expand_scale1 = float(expand_scale[0])
                    expand_scale2 = float(expand_scale[1])
                    new_mesh = self.openmesh2clothmesh(new_mesh, expand_scale1, expand_scale2, hollow_flag=True, comp_num=comp_num)
                centroid = (new_mesh.vertices.max(0) + new_mesh.vertices.min(0)) / 2
                new_mesh.vertices = new_mesh.vertices - centroid
                scale = np.abs(new_mesh.vertices).max()
                new_mesh.vertices = new_mesh.vertices / scale * self.cfg.shape_init_params
                self.scale = torch.ones(3).float().to(self.device) * (1 / self.cfg.shape_init_params * scale)
                self.trans = torch.from_numpy(centroid).float().to(self.device)
                sdf = SDF(new_mesh.vertices, new_mesh.faces)
                self.temp_mesh = new_mesh

                def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                    # add a negative signed here
                    # as in pysdf the inside of the shape has positive signed distance
                    return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                        points_rand
                    )[..., None]
                get_gt_sdf = func
        else:   
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(4000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            if sdf is None:
                points_rand = (
                    torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                )
            else:
                pt_num = 20000
                bbox_points = torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                surface_points = torch.from_numpy(sdf.sample_surface(pt_num) + np.random.normal(0, 0.05, (pt_num, 3))).float().to(self.device)
                points_rand = torch.cat((bbox_points, surface_points), dim=0)
                points_rand = points_rand[torch.randperm(points_rand.shape[0])]
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()
        del optim

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_temp_loss(self, sample_mode="random+bbox+surface", loss_mode="in", pt_num=50000):
        assert self.temp_mesh is not None
        assert self.temp_mesh_sdf is not None
        sample_pts = []
        if "random" in sample_mode:
            sample_pts.append(torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0)
        if "bbox" in sample_mode:
            center = torch.from_numpy((self.temp_mesh.vertices.max(0) + self.temp_mesh.vertices.min(0)) / 2).float().to(self.device)
            scale = torch.from_numpy((self.temp_mesh.vertices.max(0) - self.temp_mesh.vertices.min(0))).float().to(self.device)
            sample_pts.append((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * scale + center)
        if "surface" in sample_mode:
            sample_pts.append(torch.from_numpy(self.temp_mesh_sdf.sample_surface(pt_num) + np.random.normal(0, 0.01, (pt_num, 3))).float().to(self.device))
        sample_pts = torch.cat(sample_pts, dim=0)

        if "bbox" in loss_mode:
            smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
            tmp_points_list = []
            if "bottom" in self.cfg.shape_init:
                crotch_pts = smplx_pts[5620]
                bbox_center = torch.tensor([crotch_pts[0], crotch_pts[1], crotch_pts[2] - 0.21]).to(self.device)
                bbox_size = torch.tensor([0.2, 0.01, 0.4]).to(self.device)
                tmp_points = (((torch.rand((50000, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center) - self.trans) / self.scale
                tmp_max, tmp_min = tmp_points.max(0)[0], tmp_points.min(0)[0]
                in_flag = torch.logical_and(torch.all(sample_pts <= tmp_max, dim=1), torch.all(sample_pts >= tmp_min, dim=1))
                sample_pts = sample_pts[~in_flag, :]
                tmp_points_list.append(tmp_points)

            if "top" in self.cfg.shape_init:
                underarm_pts = smplx_pts[3313]
                bbox_center = torch.tensor([underarm_pts[0], underarm_pts[1], underarm_pts[2] - 0.11]).to(self.device)
                bbox_size = torch.tensor([0.2, 0.01, 0.2]).to(self.device)
                tmp_points = (((torch.rand((30000, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center) - self.trans) / self.scale
                tmp_max, tmp_min = tmp_points.max(0)[0], tmp_points.min(0)[0]
                in_flag = torch.logical_and(torch.all(sample_pts <= tmp_max, dim=1), torch.all(sample_pts >= tmp_min, dim=1))
                sample_pts = sample_pts[~in_flag, :]
                tmp_points_list.append(tmp_points)

                underarm_pts = smplx_pts[6073]
                bbox_center = torch.tensor([underarm_pts[0], underarm_pts[1], underarm_pts[2] - 0.11]).to(self.device)
                bbox_size = torch.tensor([0.2, 0.01, 0.2]).to(self.device)
                tmp_points = (((torch.rand((30000, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center) - self.trans) / self.scale
                tmp_max, tmp_min = tmp_points.max(0)[0], tmp_points.min(0)[0]
                in_flag = torch.logical_and(torch.all(sample_pts <= tmp_max, dim=1), torch.all(sample_pts >= tmp_min, dim=1))
                sample_pts = sample_pts[~in_flag, :]
                tmp_points_list.append(tmp_points)

        sample_pts = sample_pts[torch.randperm(sample_pts.shape[0])]
        sdf_gt = torch.from_numpy(-self.temp_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]

        if "mse" in loss_mode:
            sdf_pred = self.forward_sdf(sample_pts)
            loss = F.mse_loss(sdf_pred, sdf_gt, reduction='sum')
        elif "in" in loss_mode:
            in_flag = torch.where(sdf_gt < 0)[0]
            sample_pts = sample_pts[in_flag, :]
            sdf_pred = self.forward_sdf(sample_pts)
            # epsilon = torch.ones_like(sdf_pred).to(self.device) * 0
            # zero_tensor = torch.zeros_like(sdf_pred).to(self.device)
            # loss = torch.max(zero_tensor, sdf_pred - epsilon).sum()
            loss = F.relu(sdf_pred).sum()

        if "bbox" in loss_mode and len(tmp_points_list) > 0:
            tmp_points = torch.cat(tmp_points_list, dim=0)
            tmp_points = tmp_points[torch.randperm(tmp_points.shape[0])]
            tmp_sdf_pred = self.forward_sdf(tmp_points)
            loss += F.relu( - tmp_sdf_pred).sum()

        return loss

    def get_msdf_temp_loss(self, sample_mode="surface", loss_mode="in", pt_num=100000):
        assert self.temp_cyl_mesh is not None
        assert self.temp_cyl_mesh_sdf is not None
        sample_pts = []
        if "random" in sample_mode:
            sample_pts.append(torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0)
        if "bbox" in sample_mode:
            center = torch.from_numpy((self.temp_cyl_mesh.vertices.max(0) + self.temp_cyl_mesh.vertices.min(0)) / 2).float().to(self.device)
            scale = torch.from_numpy((self.temp_cyl_mesh.vertices.max(0) - self.temp_cyl_mesh.vertices.min(0))).float().to(self.device)
            sample_pts.append((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * scale + center)
        if "surface" in sample_mode:
            sample_pts.append(torch.from_numpy(self.temp_cyl_mesh_sdf.sample_surface(pt_num) + np.random.normal(0, 0.01, (pt_num, 3))).float().to(self.device))
        sample_pts = torch.cat(sample_pts, dim=0)
        sample_pts = sample_pts[torch.randperm(sample_pts.shape[0])]
        msdf_gt = torch.from_numpy(-self.temp_cyl_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]
        if loss_mode == "mse":
            msdf_pred = self.forward_msdf(sample_pts)
            loss = F.mse_loss(msdf_pred, msdf_gt, reduction='sum')
        elif loss_mode == "in":
            in_flag = torch.where(msdf_gt < 0)[0]
            sample_pts = sample_pts[in_flag, :]
            msdf_pred = self.forward_msdf(sample_pts)
            # epsilon = torch.ones_like(sdf_pred).to(self.device) * 0
            # zero_tensor = torch.zeros_like(sdf_pred).to(self.device)
            # loss = torch.max(zero_tensor, sdf_pred - epsilon).sum()
            loss = F.relu(msdf_pred).sum()

        return loss

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def get_shifted_msdf(
        self, points: Float[Tensor, "*N Di"], msdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        return msdf

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf}

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (
                        0.5
                        * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        sdf = self.sdf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_msdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        msdf = self.msdf_network(
            self.msdf_encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        msdf = self.get_shifted_msdf(points_unscaled, msdf)
        return msdf

    def forward_field(
        self, points: Float[Tensor, "*N Di"], mode="sdf+msdf"
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)

        sdf_loss: Optional[Float[Tensor, "*N 1"]] = None

        if "msdf" in mode:
            if self.cfg.msdf_type == "explicit":
                msdf = self.msdf[:points.reshape(-1, self.cfg.n_input_dims).shape[0]]
            elif self.cfg.msdf_type == "implicit":
                enc = self.msdf_encoding(points.reshape(-1, self.cfg.n_input_dims))
                msdf = self.msdf_network(enc).reshape(*points.shape[:-1], 1)
                msdf = self.get_shifted_msdf(points_unscaled, msdf)
            assert sdf.shape == msdf.shape
            return sdf, msdf, deformation, sdf_loss
        else:
            return sdf, deformation, sdf_loss

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def _initilize_explicit_msdf(self, mode="sphere"):
        if mode == "sphere":
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - 0.3
            with torch.no_grad():
                tmp_msdf = func(scale_tensor(self.isosurface_helper.grid_vertices, self.isosurface_helper.points_range, self.bbox))
                self.msdf.data[:self.isosurface_helper.grid_vertices.shape[0]] = tmp_msdf
        elif mode == "mesh":
            open_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            hole_v_idx_list = find_open_holes(open_mesh)
            print(f"hole_num: {len(hole_v_idx_list)}")
            cylinder_list = []
            for hole_idx in range(len(hole_v_idx_list)):
                hole_v_idx = hole_v_idx_list[hole_idx]
                hole_v_pos = open_mesh.vertices[hole_v_idx]
                cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos)
                cylinder_list.append(cylinder)
            cylinder = trimesh.util.concatenate(cylinder_list)
            cylinder.export(f"{self.cfg.test_save_path}/obj/init_mesh/cylinder.obj")
            cylinder.vertices = (cylinder.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy()
            cylinder_sdf = SDF(cylinder.vertices, cylinder.faces)

            self.temp_cyl_mesh = cylinder
            self.temp_cyl_mesh_sdf = cylinder_sdf

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-cylinder_sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            with torch.no_grad():
                tmp_msdf = func(scale_tensor(self.isosurface_helper.grid_vertices, self.isosurface_helper.points_range, self.bbox))
                self.msdf.data[:self.isosurface_helper.grid_vertices.shape[0]] = tmp_msdf

    def _initilize_implicit_msdf(self, mode="sphere"):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm
        if mode == "sphere":
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - 0.3
            get_gt_sdf = func
            for _ in tqdm(
                range(4000),
                desc=f"Initializing mSDF :",
                disable=get_rank() != 0,
            ):
                points_rand = (
                    torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                )
                sdf_gt = get_gt_sdf(points_rand)
                sdf_pred = self.forward_msdf(points_rand)
                loss = F.mse_loss(sdf_pred, sdf_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()
            del optim
        elif mode == "mesh":
            open_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
            hole_v_idx_list = find_open_holes(open_mesh)
            print(f"hole_num: {len(hole_v_idx_list)}")
            cylinder_list = []
            if "top" in self.cfg.shape_init:
                tmp_idx = -1
                tmp_max = -100
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    if hole_v_pos.mean(0)[2] > tmp_max:
                        tmp_max = hole_v_pos.mean(0)[2]
                        tmp_idx = hole_idx
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    if hole_idx == tmp_idx:
                        if "hot_top" in self.cfg.shape_init or "tank_top" in self.cfg.shape_init:
                            cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos, margin=0.08, round_flag=False)
                        else:
                            cylinder = fit_plane_and_create_custom_cylinder(hole_v_pos, margin=-0.05)
                    else:
                        cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos)
                    cylinder_list.append(cylinder)
            else:
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos)
                    cylinder_list.append(cylinder)
            cylinder = trimesh.util.concatenate(cylinder_list)
            cylinder.export(f"{self.cfg.test_save_path}/obj/init_mesh/cylinder.obj")
            tmp_cylinder_sdf = SDF(cylinder.vertices, cylinder.faces)
            tmp_sdf = -tmp_cylinder_sdf(close_mesh.vertices)
            tmp_mask = (tmp_sdf > 0)
            inside_faces = [f for f in close_mesh.faces if np.all(tmp_mask[f])]
            used_vertices = np.unique(np.hstack(inside_faces))
            new_vertices = close_mesh.vertices[used_vertices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
            new_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale_minus_cylinder.obj")

            cylinder.vertices = (cylinder.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy()
            cylinder_sdf = SDF(cylinder.vertices, cylinder.faces)

            self.temp_cyl_mesh = cylinder
            self.temp_cyl_mesh_sdf = cylinder_sdf

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-cylinder_sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]
            get_gt_sdf = func
            
            for _ in tqdm(
                range(4000),
                desc=f"Initializing mSDF :",
                disable=get_rank() != 0,
            ):
                pt_num = 20000
                bbox_points = torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                surface_points = torch.from_numpy(cylinder_sdf.sample_surface(pt_num) + np.random.normal(0, 0.01, (pt_num, 3))).float().to(self.device)
                points_rand = torch.cat((bbox_points, surface_points), dim=0)
                points_rand = points_rand[torch.randperm(points_rand.shape[0])]

                msdf_gt = get_gt_sdf(points_rand)
                msdf_pred = self.forward_msdf(points_rand)
                loss = F.mse_loss(msdf_pred, msdf_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()
            del optim
        elif mode == "surface":
            def nearest_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                assert x.device == y.device
                delta = x.unsqueeze(1) - y.unsqueeze(0)
                dist_sq = torch.sum(delta**2, dim=-1)
                min_dist_sq, _ = torch.min(dist_sq, dim=1)
                min_dist = torch.sqrt(torch.clamp(min_dist_sq, min=1e-16))
                return min_dist

            open_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
            tmp_ms = ml.MeshSet()
            tmp_ms.clear()
            tmp_ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            tmp_ms.apply_filter('compute_selection_from_mesh_border')
            v_selection = tmp_ms.current_mesh().vertex_selection_array()
            hole_v = torch.from_numpy(open_mesh.vertices[v_selection]).float().to(self.device)
            hole_v_idx_list = find_open_holes(open_mesh)
            print(f"hole_num: {len(hole_v_idx_list)}")
            cylinder_list = []
            if "top" in self.cfg.shape_init:
                tmp_idx = -1
                tmp_max = -100
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    if hole_v_pos.mean(0)[2] > tmp_max:
                        tmp_max = hole_v_pos.mean(0)[2]
                        tmp_idx = hole_idx
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    if hole_idx == tmp_idx:
                        cylinder = fit_plane_and_create_custom_cylinder(hole_v_pos, margin=-0.05)
                    else:
                        cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos)
                    cylinder_list.append(cylinder)
            else:
                for hole_idx in range(len(hole_v_idx_list)):
                    hole_v_idx = hole_v_idx_list[hole_idx]
                    hole_v_pos = open_mesh.vertices[hole_v_idx]
                    cylinder, n, t = fit_plane_and_create_elliptical_cylinder(hole_v_pos)
                    cylinder_list.append(cylinder)
            cylinder = trimesh.util.concatenate(cylinder_list)
            cylinder.export(f"{self.cfg.test_save_path}/obj/init_mesh/cylinder.obj")
            tmp_cylinder_sdf = SDF(cylinder.vertices, cylinder.faces)
            tmp_sdf = -tmp_cylinder_sdf(close_mesh.vertices)
            tmp_mask = (tmp_sdf > 0)
            inside_faces = [f for f in close_mesh.faces if np.all(tmp_mask[f])]
            used_vertices = np.unique(np.hstack(inside_faces))
            new_vertices = close_mesh.vertices[used_vertices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
            out_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
            out_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale_out_mesh.obj")

            tmp_mask = (tmp_sdf <= 0)
            inside_faces = [f for f in close_mesh.faces if np.all(tmp_mask[f])]
            used_vertices = np.unique(np.hstack(inside_faces))
            new_vertices = close_mesh.vertices[used_vertices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
            in_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
            in_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale_in_mesh.obj")

            out_mesh.vertices = (out_mesh.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy()
            in_mesh.vertices = (in_mesh.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy()
            close_mesh.vertices = (close_mesh.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy()
            hole_v = (hole_v - self.trans.detach()) / self.scale.detach()
            tmp_cylinder_sdf = SDF((cylinder.vertices - self.trans.detach().cpu().numpy()) / self.scale.detach().cpu().numpy(), cylinder.faces)              
            close_mesh_sdf = SDF(close_mesh.vertices, close_mesh.faces)

            for _ in tqdm(
                range(4000),
                desc=f"Initializing mSDF :",
                disable=get_rank() != 0,
            ):
                surface_points = torch.from_numpy(close_mesh_sdf.sample_surface(20000)).float().to(self.device)
                tmp_sdf = -tmp_cylinder_sdf(surface_points.detach().cpu().numpy())
                GT = nearest_distance(surface_points, hole_v)
                GT[tmp_sdf < 0] = - GT[tmp_sdf < 0]
                temp_data = torch.cat((surface_points, GT[:, None]), dim=1)
                temp_data = temp_data[torch.randperm(temp_data.shape[0])]
                msdf_pred = self.forward_msdf(temp_data[:, :3])
                loss = F.mse_loss(msdf_pred, temp_data[:, 3:])
                optim.zero_grad()
                loss.backward()
                optim.step()
            del optim
            
    def _initilize_isosurface_helper(self, mesh = None, save_dir_path = None):
        if self.cfg.isosurface and self.isosurface_helper is None:
            if self.cfg.isosurface_method == "mc-cpu":
                if self.cfg.isosurface_resolution.split(":")[0] == "cube":
                    self.isosurface_helper = MarchingCubeCPUHelper(
                        int(self.cfg.isosurface_resolution.split(":")[1])
                    ).to(self.device)
                else:
                    raise AttributeError(
                        "MarchingCubeCPUHelper only supports cube resolution"
                    )
            elif self.cfg.isosurface_method == "mt":
                if self.cfg.isosurface_resolution.split(":")[0] == "cube":
                    tmp_res = int(self.cfg.isosurface_resolution.split(":")[1])
                    self.isosurface_helper = MarchingTetrahedraHelper(
                    tmp_res,
                    f"load/tets/{tmp_res}_tets.npz",
                    ).to(self.device)
                elif self.cfg.isosurface_resolution.split(":")[0] == "shell":
                    tmp_para = self.cfg.isosurface_resolution.split(":")[1:]
                    self.isosurface_helper = MarchingTetrahedraShellHelper(
                        mesh, float(tmp_para[0]), float(tmp_para[1]), float(tmp_para[2]), save_dir_path
                    ).to(self.device)
            elif self.cfg.isosurface_method == "gshell":
                if self.cfg.isosurface_resolution.split(":")[0] == "cube":
                    tmp_res = int(self.cfg.isosurface_resolution.split(":")[1])
                    self.isosurface_helper = GshellHelper(resolution=tmp_res, tets_path=f"load/tets/{tmp_res}_tets.npz"
                    ).to(self.device)
                elif self.cfg.isosurface_resolution.split(":")[0] == "shell":
                    tmp_para = self.cfg.isosurface_resolution.split(":")[1:]
                    self.isosurface_helper = GshellHelper(
                    mesh=mesh, tet_shell_offset=float(tmp_para[0]), tet_shell_decimate=float(tmp_para[1]), tet_grid_volume=float(tmp_para[2]), save_dir_path=save_dir_path
                    ).to(self.device)
            else:
                raise AttributeError(
                    "Unknown isosurface method {self.cfg.isosurface_method}"
                )

    def _isosurface_wt(self, bbox: Float[Tensor, "2 3"], fine_stage: bool = False) -> Mesh:
        def batch_func(x):
            # scale to bbox as the input vertices are in [0, 1]
            field, deformation, sdf_loss = self.forward_field(
                scale_tensor(
                    x.to(bbox.device), self.isosurface_helper.points_range, bbox
                ), "sdf"
            )
            field = field.to(
                x.device
            )  # move to the same device as the input (could be CPU)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation, sdf_loss

        assert self.isosurface_helper is not None

        field, deformation, sdf_loss = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
        )

        threshold: float

        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            threestudio.info(
                f"Automatically determined isosurface threshold: {threshold}"
            )
        else:
            raise TypeError(
                f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}"
            )

        level = self.forward_level(field, threshold)
        mesh: Mesh = self.isosurface_helper(level, deformation=deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        mesh.add_extra("bbox", bbox)

        if self.cfg.isosurface_remove_outliers:
            # remove outliers components with small number of faces
            # only enabled when the mesh is not differentiable
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)

        return mesh

    def _isosurface_nonwt(self, bbox: Float[Tensor, "2 3"], output_watertight_template_flag=True) -> Mesh:
        def batch_func(x):
            # scale to bbox as the input vertices are in [0, 1]
            sdf, msdf, deformation, sdf_loss = self.forward_field(
                scale_tensor(
                    x.to(bbox.device), self.isosurface_helper.points_range, bbox
                ), "sdf+msdf"
            )
            sdf = sdf.to(
                x.device
            )
            msdf = msdf.to(
                x.device
            )
            # move to the same device as the input (could be CPU)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return sdf, msdf, deformation, sdf_loss

        assert self.isosurface_helper is not None

        sdf, msdf, deformation, sdf_loss = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
        )

        mesh, extra = self.isosurface_helper.forward_gshell(sdf, msdf, deformation, output_watertight_template_flag)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        mesh.add_extra("bbox", bbox)

        if output_watertight_template_flag:
            extra["mesh_watertight"].v_pos = scale_tensor(
                extra["mesh_watertight"].v_pos, self.isosurface_helper.points_range, bbox
            )  # scale to bbox as the grid vertices are in [0, 1]
            extra["mesh_watertight"].add_extra("bbox", bbox)

        return mesh, extra

    def isosurface(self, mode="nonwt") -> Mesh:
        self._initilize_isosurface_helper()
        if mode == "wt":
            mesh = self._isosurface_wt(self.bbox)
            return mesh
        elif mode == "nonwt":
            mesh, extra = self._isosurface_nonwt(self.bbox, output_watertight_template_flag=False)
            return mesh, extra
        elif mode == "nonwt+wt":
            mesh, extra = self._isosurface_nonwt(self.bbox, output_watertight_template_flag=True)
            return mesh, extra

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        import trimesh
        if global_step >= (self.cfg.start_sdf_loss_step + 1) and self.cached_sdf is None:  
            mesh_v_pos = np.load(f'{self.cfg.test_save_path}/mesh_v_pos.npy')
            mesh_t_pos_idx = np.load(f'{self.cfg.test_save_path}/mesh_t_pos_idx.npy')
            cached_mesh = trimesh.Trimesh(
                vertices=mesh_v_pos,
                faces=mesh_t_pos_idx,
            )
            self.cached_sdf = SDF(cached_mesh.vertices, cached_mesh.faces)
        if self.temp_mesh is not None:
            if global_step % self.cfg.update_temp_loss_step == 0:
                if global_step == 0:
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    os.makedirs(f"{self.cfg.test_save_path}/obj/temp", exist_ok=True)
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/temp/temp_{global_step}.obj")
                else:
                    tmp_mesh = self.isosurface(mode="wt")
                    self.temp_mesh = trimesh.Trimesh(
                            vertices=tmp_mesh.v_pos.detach().cpu().numpy(),
                            faces=tmp_mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/temp/temp_{global_step}.obj")

        if self.cfg.isosurface_resolution.split(":")[0] == "cube":
            tmp_res = int(self.cfg.isosurface_resolution.split(":")[1])
            resolution = tmp_res
            for idx, step in enumerate(self.cfg.progressive_resolution.step):
                if global_step < step:
                    break
                resolution = int(self.cfg.progressive_resolution.resolution[idx])
            if resolution > tmp_res:
                self.cfg.isosurface_resolution = f"cube:{resolution}" 
                self.isosurface_helper = None
                self._initilize_isosurface_helper()
        # if self.cfg.progressive_resolution_steps is not None:
        #     if global_step >= self.cfg.progressive_resolution_steps[0] and self.cfg.isosurface_resolution < 256:
        #         self.cfg.isosurface_resolution = 256
        #         self.isosurface_helper = None
        #         self._initilize_isosurface_helper()
        #     if global_step >= self.cfg.progressive_resolution_steps[1] and self.cfg.isosurface_resolution < 512:
        #         self.cfg.isosurface_resolution = 512
        #         self.isosurface_helper = None
        #         self._initilize_isosurface_helper()

        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )
