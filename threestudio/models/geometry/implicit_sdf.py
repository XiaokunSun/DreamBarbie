import os
import random
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from pysdf import SDF
import trimesh
from threestudio.utils.utils import compute_normal
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import pickle

def save_pc(pc, save_path, color=None):
    obj_str = ""
    for i in range(pc.shape[0]):
        if color is None:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]}"
        else:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]} {color[i][0]} {color[i][1]} {color[i][2]}"
        obj_str += "\n"
    obj_str += "\n"
    with open(save_path, "w") as f:
        f.write(obj_str)
    return

@threestudio.register("implicit-sdf")
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
        progressive_resolution_steps: Optional[int] = None

        # new
        test_save_path: str = "./.threestudio_cache"
        smplx_path: str = "./load/smplx_models"
        gender: str = "neutral"
        start_smplx_loss_step: int = 0
        update_smplx_loss_step: int = 1000
        opt_smplx_step_num: int = 1000

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

        if self.cfg.shape_init.startswith("opt_smplx:"):
            tmp_strlist = self.cfg.shape_init.split(":")
            self.opt_smplx_flag = True
            self.canpose = tmp_strlist[1]
            self.num_betas = int(tmp_strlist[2])
            self.dim_offsets = int(tmp_strlist[3])
            self.evolving_flag = (tmp_strlist[4] == "True")
            # self.betas_lr = float(tmp_strlist[5])
            # self.evolving_offset_flag = (tmp_strlist[6] == "True")
            self.smplx_model = smplx.create(
                    self.cfg.smplx_path,
                    model_type="smplx",
                    gender=self.cfg.gender,
                    use_face_contour=False,
                    num_betas=self.num_betas,
                    num_expression_coeffs=10,
                    ext="npz",
                    use_pca=False,  # explicitly control hand pose
                    flat_hand_mean=True,  # use a flatten hand default pose
                )
            with open(f'{self.cfg.smplx_path}/smplx/smplx_watertight.pkl', 'rb') as f:
                tmp_data = pickle.load(f, encoding='latin1')
            self.smplx_model.faces = tmp_data["smpl_F"].numpy()
            N = self.smplx_model.lbs_weights.shape[0]
            self.register_parameter(
                'betas', nn.Parameter(torch.zeros(self.num_betas), requires_grad=True))
            self.register_parameter(
                'v_offsets', nn.Parameter(torch.zeros(N, self.dim_offsets), requires_grad=True))
            body_pose = np.zeros((21, 3), dtype=np.float32)
            if self.canpose == "apose":
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
            elif self.canpose == 'tpose':
                pass
            elif self.canpose == "dapose":
                body_pose[0, 1] = 0.2
                body_pose[0, 2] = 0.2
                body_pose[1, 1] = -0.2
                body_pose[1, 2] = -0.2
                body_pose[15, 2] = -0.4
                body_pose[16, 2] = 0.4
            elif self.canpose == "dapose1":
                body_pose[0, 1] = 0.2
                body_pose[0, 2] = 0.2
                body_pose[1, 1] = -0.2
                body_pose[1, 2] = -0.2
                body_pose[15, 2] = -0.7
                body_pose[16, 2] = 0.7
            elif self.canpose == 'dapose2':
                body_pose[0, 1] = 0.2
                body_pose[0, 2] = 0.2
                body_pose[1, 1] = -0.2
                body_pose[1, 2] = -0.2
                body_pose[15, 2] = -0.8
                body_pose[16, 2] = 0.8
            self.body_pose = torch.from_numpy(body_pose).float().to(self.device)
            self.jaw_pose = torch.tensor([0, 0, 0], dtype=torch.float32).float().to(self.device)
        else:
            self.opt_smplx_flag = False
            self.evolving_flag = False
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
        self.smplx_mesh = None
        self.smplx_mesh_sdf = None
        self.temp_mesh = None
        self.temp_mesh_sdf = None
        self.part_info_dict = None
        self.body_mask = None
        self.temp_mask = None

    def initialize_shape(self, verts=None, faces=None, system_cfg=None) -> None:
        sdf = None

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
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
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
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # adjust the position of mesh
            if "full_body" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.3
            elif "half_body" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.1
            elif "head_only" in mesh_path:
                mesh.vertices[:,2] = mesh.vertices[:,2] + 0.15
            elif "t-pose" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.4
            elif "pose" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.4

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

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("smplx:"):
            assert isinstance(self.cfg.shape_init_params, float)
            shape_init_strlist = self.cfg.shape_init.split(":")
            body_pose = np.zeros((21, 3), dtype=np.float32)
            if shape_init_strlist[1] == "apose":
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
            elif shape_init_strlist[1] == 'tpose':
                pass
            elif shape_init_strlist[1] == 'dapose':
                body_pose[0, 1] = 0.2
                body_pose[0, 2] = 0.2
                body_pose[1, 1] = -0.2
                body_pose[1, 2] = -0.2
                body_pose[15, 2] = -0.4
                body_pose[16, 2] = 0.4
            elif shape_init_strlist[1] == 'dapose1':
                body_pose[0, 1] = 0.2
                body_pose[0, 2] = 0.2
                body_pose[1, 1] = -0.2
                body_pose[1, 2] = -0.2
                body_pose[15, 2] = -0.7
                body_pose[16, 2] = 0.7
            elif shape_init_strlist[1] == 'dapose2':
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
            if shape_init_strlist[1] == "apose":
                smplx_mesh.vertices[:,1] = smplx_mesh.vertices[:,1] + 0.3
            elif shape_init_strlist[1] == "dapose":
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
            scale = np.abs(smplx_mesh.vertices).max()
            smplx_mesh.vertices = smplx_mesh.vertices / scale * self.cfg.shape_init_params
            smplx_mesh.vertices = np.dot(mesh2std, smplx_mesh.vertices.T).T
            
            sdf = SDF(smplx_mesh.vertices, smplx_mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

            self.smplx_mesh = smplx_mesh
            self.temp_mesh = smplx_mesh
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

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def initialize_shape_from_smplx(self, smplx_mesh, global_step):
        sdf = SDF(smplx_mesh.vertices, smplx_mesh.faces)
        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]
        get_gt_sdf = func
        self.smplx_mesh = smplx_mesh
        self.temp_mesh = smplx_mesh
        self.smplx_mesh_sdf = SDF(self.smplx_mesh.vertices, self.smplx_mesh.faces)
        self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
        os.makedirs(f"{self.cfg.test_save_path}/obj/smplx", exist_ok=True)
        self.smplx_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/smplx.obj")
        self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/temp_{global_step}.obj")
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
                rand_points = torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                surface_points = torch.from_numpy(sdf.sample_surface(pt_num) + np.random.normal(0, 0.05, (pt_num, 3))).float().to(self.device)
                center = torch.from_numpy((self.smplx_mesh.vertices.max(0) + self.smplx_mesh.vertices.min(0)) / 2).float().to(self.device)
                scale = torch.from_numpy((self.smplx_mesh.vertices.max(0) - self.smplx_mesh.vertices.min(0))).float().to(self.device)
                bbox_points = (torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * scale + center

                LBS_weights = self.smplx_model.lbs_weights.detach().to(self.device)
                LBS_weights = LBS_weights[:self.smplx_mesh.vertices.shape[0], :]
                smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
                LBS_argmax = torch.argmax(LBS_weights, dim=-1)
                head_mask = torch.zeros_like(LBS_argmax).to(self.device)
                for tmp_idx in [15, 22]:
                    head_mask = torch.logical_or(head_mask, LBS_argmax == tmp_idx)
                head_pts = smplx_pts[head_mask]
                head_bbox_center = (head_pts.min(0)[0] + head_pts.max(0)[0]) / 2
                head_bbox_scale = (head_pts.max(0)[0] - head_pts.min(0)[0])
                head_points = (torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * head_bbox_scale + head_bbox_center
                points_rand = torch.cat((rand_points, surface_points, bbox_points, head_points), dim=0)
                points_rand = points_rand[torch.randperm(points_rand.shape[0])]

            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

    def get_opt_smplx_mesh(self, offsets_flag=True, filter_flag=True):
        smplx_output = self.smplx_model(betas=self.betas.unsqueeze(0), body_pose=self.body_pose.unsqueeze(0), jaw_pose=self.jaw_pose.unsqueeze(0), return_verts=True)
        smplx_vertices = smplx_output.vertices[0]
        smplx_faces = self.smplx_model.faces

        centroid = smplx_vertices.mean(0)
        smplx_vertices = smplx_vertices - centroid

        # adjust the position of mesh
        if self.canpose == "apose":
            smplx_vertices[:,1] = smplx_vertices[:,1] + 0.3
        elif self.canpose == "dapose":
            smplx_vertices[:,1] = smplx_vertices[:,1] + 0.4
        else:
            smplx_vertices[:,1] = smplx_vertices[:,1] + 0.4

        # align to up-z and front-x
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": torch.tensor([1, 0, 0]).to(self.device).float(),
            "+y": torch.tensor([0, 1, 0]).to(self.device).float(),
            "+z": torch.tensor([0, 0, 1]).to(self.device).float(),
            "-x": torch.tensor([-1, 0, 0]).to(self.device).float(),
            "-y": torch.tensor([0, -1, 0]).to(self.device).float(),
            "-z": torch.tensor([0, 0, -1]).to(self.device).float(),
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
        y_ = torch.cross(z_, x_)
        std2mesh = torch.stack([x_, y_, z_], axis=0).T
        mesh2std = torch.linalg.inv(std2mesh)

        # scaling
        scale = torch.abs(smplx_vertices).max().detach()
        smplx_vertices = smplx_vertices / scale * self.cfg.shape_init_params
        smplx_vertices = torch.matmul(mesh2std, smplx_vertices.T).T 

        smplx_vertices_normal = compute_normal(smplx_vertices, smplx_faces.astype(np.int64), smplx_vertices.device)
        
        if self.body_mask == None:
            LBS_weights = self.smplx_model.lbs_weights.detach().to(self.device)
            LBS_argmax = torch.argmax(LBS_weights, dim=-1)
            body_mask = torch.zeros_like(LBS_argmax).to(self.device)
            # for part_idx in ([7, 8, 10, 11, 15, 20, 21, 22, 23, 24] + list(range(25, 55))):

            for part_idx in ([7, 8, 10, 11, 20, 21, 23, 24] + list(range(25, 55))):
                body_mask = torch.logical_or(body_mask, LBS_argmax == part_idx)
            tmp_idx = np.load(f'{self.cfg.smplx_path}/smplx/smplx_face_ears_noeyeballs_idx.npy')
            body_mask[torch.from_numpy(tmp_idx).long().to(body_mask.device)] = True
            self.body_mask = ~body_mask

        if offsets_flag:
            if filter_flag:
                if self.v_offsets.shape[1] == 1:
                    smplx_vertices[self.body_mask] += smplx_vertices_normal[0][self.body_mask] * self.v_offsets[self.body_mask]
                else:
                    smplx_vertices[self.body_mask] += self.v_offsets[self.body_mask]
            else:
                if self.v_offsets.shape[1] == 1:
                    smplx_vertices += smplx_vertices_normal[0] * self.v_offsets
                else:
                    smplx_vertices += self.v_offsets

        # return smplx_vertices.contiguous(), torch.from_numpy(smplx_faces.astype(np.int64)).long().to(self.device).contiguous(), smplx_vertices_normal.contiguous()
        return Mesh(v_pos=smplx_vertices.contiguous(), t_pos_idx=torch.from_numpy(smplx_faces.astype(np.int64)).long().to(self.device).contiguous())


    def get_smplx_local_loss(self, sample_mode="bbox+surface", sample_part="rh+lh+rf+lf", loss_mode="mse", pt_num=20000):
        assert self.smplx_mesh is not None
        assert self.smplx_mesh_sdf is not None
        sample_pts = []
        if self.part_info_dict is None:
            self.part_info_dict = {}
            LBS_weights = self.smplx_model.lbs_weights.detach().to(self.device)
            LBS_weights = LBS_weights[:self.smplx_mesh.vertices.shape[0], :]
            smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
            LBS_argmax = torch.argmax(LBS_weights, dim=-1)
            part_dict = {"head":[15, 22], "lh":[20] + list(range(25, 40)), "rh":[21] + list(range(40, 55)), "lf":[7,10], "rf":[8,11]}
            for part in sample_part.split("+"):
                part_mask = torch.zeros_like(LBS_argmax).to(self.device)
                for part_idx in part_dict[part]:
                    part_mask = torch.logical_or(part_mask, LBS_argmax == part_idx)
                part_pts = smplx_pts[part_mask]
                part_bbox_center = (part_pts.min(0)[0] + part_pts.max(0)[0]) / 2
                part_bbox_scale = (part_pts.max(0)[0] - part_pts.min(0)[0])
                self.part_info_dict.update({f"{part}_mask":part_mask, f"{part}_pts":part_pts, f"{part}_bbox_center":part_bbox_center, f"{part}_bbox_scale":part_bbox_scale})
        for part in sample_part.split("+"):
            part_pts = self.part_info_dict[f"{part}_pts"]
            part_bbox_center = self.part_info_dict[f"{part}_bbox_center"]
            part_bbox_scale = self.part_info_dict[f"{part}_bbox_scale"]
            max_scale = part_bbox_scale.max().cpu().item()
            if "bbox" in sample_mode:
                sample_pts.append((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * part_bbox_scale + part_bbox_center)
            if "surface" in sample_mode:
                sample_pts.append(torch.from_numpy(np.random.normal(0, max_scale / 1.8 * 0.01, (pt_num, 3))).float().to(self.device) + torch.index_select(part_pts, 0, torch.randint(0, part_pts.shape[0], (pt_num, )).to(self.device)))
        sample_pts = torch.cat(sample_pts, dim=0)
        sample_pts = sample_pts[torch.randperm(sample_pts.shape[0])]
        sdf_gt = torch.from_numpy(-self.smplx_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]
        if loss_mode == "mse":
            sdf_pred = self.forward_sdf(sample_pts)
            loss = F.mse_loss(sdf_pred, sdf_gt, reduction='sum')
        elif loss_mode == "in":
            in_flag = torch.where(sdf_gt < 0)[0]
            sample_pts = sample_pts[in_flag, :]
            sdf_pred = self.forward_sdf(sample_pts)
            epsilon = torch.ones_like(sdf_pred).to(self.device) * 0
            zero_tensor = torch.zeros_like(sdf_pred).to(self.device)
            loss = torch.max(zero_tensor, sdf_pred - epsilon).sum()

        if True:
            smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
            crotch_pts = smplx_pts[5620]
            bbox_center = torch.tensor([crotch_pts[0], crotch_pts[1], crotch_pts[2] - 0.21]).to(self.device)
            bbox_size = torch.tensor([0.4, 0.02, 0.4]).to(self.device)
            tmp_points = ((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center)
            tmp_sdf_pred = self.forward_sdf(tmp_points)
            # epsilon = torch.ones_like(tmp_sdf_pred).to(self.device) * 0
            # zero_tensor = torch.zeros_like(tmp_sdf_pred).to(self.device)
            # loss += torch.min(zero_tensor, tmp_sdf_pred - epsilon).sum()
            loss += F.relu( - tmp_sdf_pred).sum()

        if True:
            smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
            underarm_pts = smplx_pts[3313]
            bbox_center = torch.tensor([underarm_pts[0], underarm_pts[1], underarm_pts[2] - 0.11]).to(self.device)
            bbox_size = torch.tensor([0.2, 0.02, 0.2]).to(self.device)
            tmp_points = ((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center)
            tmp_sdf_pred = self.forward_sdf(tmp_points)
            # epsilon = torch.ones_like(tmp_sdf_pred).to(self.device) * 0
            # zero_tensor = torch.zeros_like(tmp_sdf_pred).to(self.device)
            # loss += torch.min(zero_tensor, tmp_sdf_pred - epsilon).sum()
            loss += F.relu( - tmp_sdf_pred).sum()

            smplx_pts = torch.from_numpy(self.smplx_mesh.vertices).float().to(self.device)
            underarm_pts = smplx_pts[6073]
            bbox_center = torch.tensor([underarm_pts[0], underarm_pts[1], underarm_pts[2] - 0.11]).to(self.device)
            bbox_size = torch.tensor([0.2, 0.02, 0.2]).to(self.device)
            tmp_points = ((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * bbox_size + bbox_center)
            tmp_sdf_pred = self.forward_sdf(tmp_points)
            # epsilon = torch.ones_like(tmp_sdf_pred).to(self.device) * 0
            # zero_tensor = torch.zeros_like(tmp_sdf_pred).to(self.device)
            # loss += torch.min(zero_tensor, tmp_sdf_pred - epsilon).sum()
            loss += F.relu( - tmp_sdf_pred).sum()

        return loss

    def get_smplx_global_loss(self, sample_mode="random+bbox+surface", loss_mode="mse", pt_num=20000):
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
        sample_pts = sample_pts[torch.randperm(sample_pts.shape[0])]

        sdf_gt = torch.from_numpy(-self.temp_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]
        if loss_mode == "mse":
            sdf_pred = self.forward_sdf(sample_pts)
            loss = F.mse_loss(sdf_pred, sdf_gt, reduction='sum')
        elif loss_mode == "in":
            in_flag = torch.where(sdf_gt < 0)[0]
            sample_pts = sample_pts[in_flag, :]
            sdf_pred = self.forward_sdf(sample_pts)
            epsilon = torch.ones_like(sdf_pred).to(self.device) * 0
            zero_tensor = torch.zeros_like(sdf_pred).to(self.device)
            loss = torch.max(zero_tensor, sdf_pred - epsilon).sum()

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

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
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
        return sdf, deformation, sdf_loss

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

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
        if self.smplx_mesh is not None and self.temp_mesh is not None:
            if global_step % self.cfg.update_smplx_loss_step == 0:
                if global_step == 0:
                    self.smplx_mesh_sdf = SDF(self.smplx_mesh.vertices, self.smplx_mesh.faces)
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    os.makedirs(f"{self.cfg.test_save_path}/obj/smplx", exist_ok=True)
                    self.smplx_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/smplx.obj")
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/temp_{global_step}.obj")
                else:
                    if global_step >= self.cfg.start_smplx_loss_step:
                        if self.opt_smplx_flag and self.evolving_flag:
                            tmp_mesh = self.isosurface()
                            GT_mesh = Meshes(verts=[tmp_mesh.v_pos.detach()], faces=[tmp_mesh.t_pos_idx.detach()])
                            from tqdm import tqdm
                            w_chamfer = 100.0
                            w_edge = 1.0
                            w_normal = 0.01
                            w_laplacian = 0.1
                            # optim = torch.optim.Adam([self.betas], lr=self.betas_lr)
                            # for i in tqdm(range(100)):
                            #     optim.zero_grad()
                            #     tmp_smplx_mesh = self.get_opt_smplx_mesh(offsets_flag=False, filter_flag=False)
                            #     smplx_mesh = Meshes(verts=[tmp_smplx_mesh.v_pos], faces=[tmp_smplx_mesh.t_pos_idx])

                            #     GT_sample_pts = sample_points_from_meshes(GT_mesh, 50000)
                            #     smplx_sample_pts = sample_points_from_meshes(smplx_mesh, 50000)
                                
                            #     # We compare the two sets of pointclouds by computing (a) the chamfer loss
                            #     loss_chamfer, _ = chamfer_distance(GT_sample_pts, smplx_sample_pts)
                                
                            #     # and (b) the edge length of the predicted mesh
                            #     loss_edge = mesh_edge_loss(smplx_mesh)
                                
                            #     # mesh normal consistency
                            #     loss_normal = mesh_normal_consistency(smplx_mesh)
                                
                            #     # mesh laplacian smoothing
                            #     loss_laplacian = mesh_laplacian_smoothing(smplx_mesh, method="uniform")
                            #     # print(torch.abs(self.betas).mean().item())
                            #     # Weighted sum of the losses
                            #     loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
                            #     loss.backward()
                            #     optim.step()
                            # optim.zero_grad()

                            # if self.evolving_offset_flag:
                            optim = torch.optim.Adam([self.v_offsets], lr=1e-4)
                            for i in tqdm(range(100)):
                                optim.zero_grad()
                                tmp_smplx_mesh = self.get_opt_smplx_mesh(offsets_flag=True, filter_flag=True)
                                smplx_mesh = Meshes(verts=[tmp_smplx_mesh.v_pos], faces=[tmp_smplx_mesh.t_pos_idx])

                                GT_sample_pts = sample_points_from_meshes(GT_mesh, 50000)
                                smplx_sample_pts = sample_points_from_meshes(smplx_mesh, 50000)
                                
                                # We compare the two sets of pointclouds by computing (a) the chamfer loss
                                loss_chamfer, _ = chamfer_distance(GT_sample_pts, smplx_sample_pts)
                                
                                # and (b) the edge length of the predicted mesh
                                loss_edge = mesh_edge_loss(smplx_mesh)
                                
                                # mesh normal consistency
                                loss_normal = mesh_normal_consistency(smplx_mesh)
                                
                                # mesh laplacian smoothing
                                loss_laplacian = mesh_laplacian_smoothing(smplx_mesh, method="uniform")
                                # print(torch.abs(self.v_offsets).mean().item())
                                # Weighted sum of the losses
                                loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
                                loss.backward()
                                optim.step()
                            optim.zero_grad()

                            final_verts, final_faces = smplx_mesh.get_mesh_verts_faces(0)
                            self.temp_mesh = trimesh.Trimesh(
                                    vertices=final_verts.detach().cpu().numpy(),
                                    faces=final_faces.detach().cpu().numpy(),
                                )
                            self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                            self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/temp_{global_step}.obj")
                        else:
                            tmp_mesh = self.isosurface()
                            self.temp_mesh = trimesh.Trimesh(
                                    vertices=tmp_mesh.v_pos.detach().cpu().numpy(),
                                    faces=tmp_mesh.t_pos_idx.detach().cpu().numpy(),
                                )
                            self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                            self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/smplx/temp_{global_step}.obj")

        if self.cfg.isosurface_resolution.split(":")[0] == "cube":
            tmp_res = int(self.cfg.isosurface_resolution.split(":")[1])
            if self.cfg.progressive_resolution_steps is not None:
                if global_step >= self.cfg.progressive_resolution_steps[0] and tmp_res < 256:
                    self.cfg.isosurface_resolution = f"cube:256"
                    self.isosurface_helper = None
                    self._initilize_isosurface_helper()
                if global_step >= self.cfg.progressive_resolution_steps[1] and tmp_res < 512:
                    self.cfg.isosurface_resolution = f"cube:512"
                    self.isosurface_helper = None
                    self._initilize_isosurface_helper()

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
