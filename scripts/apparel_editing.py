
import os
import torch
import trimesh
from glob import glob
import cubvh
import numpy as np
import pymeshlab as ml
import sys
import imageio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threestudio.utils import smplx_customized
import torch
import heapq
from tqdm import tqdm
import cv2
from PIL import Image
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.rendering import *
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda"
ctx = NVDiffRasterizerContext("cuda", device)
render_data = prepare_render_data(120, device)
data = sample_render_data(render_data, 0)
human_mesh_dir_path = "./outputs/clothed_human/stage3-fine-texture/A_strong_grown_man_wearing_cargo_pants_brown_leather_boots_a_red_plaid_flannel_shirt_a_work_glove_on_his_left_hand_and_a_silver_wristwatch_on_his_right_hand" # Replace this with the path to your generated clothed human
smplx_mesh_dir_path = "./outputs/naked_human/stage1-geometry/A_strong_grown_man_in_his_underwear" # Replace this with the path to your generated naked human
save_path = "./outputs/edit_garment"
test_name = "cargo_pants" # garment name
shape_init_params = 0.9
dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
dir2vec = {
    "+x": torch.tensor([1, 0, 0]).to(device).float(),
    "+y": torch.tensor([0, 1, 0]).to(device).float(),
    "+z": torch.tensor([0, 0, 1]).to(device).float(),
    "-x": torch.tensor([-1, 0, 0]).to(device).float(),
    "-y": torch.tensor([0, -1, 0]).to(device).float(),
    "-z": torch.tensor([0, 0, -1]).to(device).float(),
}
z_, x_ = (
    dir2vec["+y"],
    dir2vec["+z"],
)
y_ = torch.cross(z_, x_)
std2mesh = torch.stack([x_, y_, z_], axis=0).T
mesh2std = torch.linalg.inv(std2mesh)

human_ckpt = torch.load(os.path.join(smplx_mesh_dir_path, "ckpts/epoch=0-step=6000.ckpt"))
betas = human_ckpt["state_dict"]["geometry.betas"].cpu()
v_offsets = human_ckpt["state_dict"]["geometry.v_offsets"].cpu()
smplx_model = smplx_customized.create(
    "./load/smplx_models",
    model_type="smplx",
    gender="neutral",
    use_face_contour=False,
    num_betas=10,
    num_expression_coeffs=50,
    ext="npz",
    use_pca=False,  # explicitly conrol hand pose
    flat_hand_mean=True,  # use a flatten hand default pose
)
smplx_lbs_weights = smplx_model.lbs_weights.to(device).float()
init_body_pose = np.zeros((21, 3), dtype=np.float32)
init_body_pose[0, 1] = 0.2
init_body_pose[0, 2] = 0.2
init_body_pose[1, 1] = -0.2
init_body_pose[1, 2] = -0.2
init_body_pose[15, 2] = -0.7
init_body_pose[16, 2] = 0.7

smplx_mesh = trimesh.load(os.path.join(smplx_mesh_dir_path, "save/cache/obj/smplx/temp_5005.obj"))
human_mesh_list = []
for mesh_path in sorted(glob(os.path.join(human_mesh_dir_path, "save/obj/result/comp_1.obj"))): # The OBJ file of the edited clothing item
    components = trimesh.load(mesh_path).split(only_watertight=False)
    if os.path.basename(mesh_path) == "comp_2.obj":
        tmp = heapq.nlargest(2, components, key=lambda comp: len(comp.faces))
    else:
        tmp = heapq.nlargest(1, components, key=lambda comp: len(comp.faces))
    tmp = trimesh.util.concatenate(tmp)
    human_mesh_list.append(tmp)
human_mesh = trimesh.util.concatenate(human_mesh_list)
smplx_mesh_BVH = cubvh.cuBVH(smplx_mesh.vertices, smplx_mesh.faces)
smplx_verts_tensor = torch.from_numpy(smplx_mesh.vertices).to(device).float()
smplx_faces_tensor = torch.from_numpy(smplx_mesh.faces).to(device).long()
human_verts_tensor = torch.from_numpy(human_mesh.vertices).to(device).float()
human_faces_tensor = torch.from_numpy(human_mesh.faces).to(device).long()
human_rgb_tensor = (torch.from_numpy(human_mesh.visual.vertex_colors) / 255).float().to(device)[:, :3]

center = human_verts_tensor.mean(0)
scale = torch.max((human_verts_tensor - center).abs())

rot_smplx_verts_tensor = torch.matmul(std2mesh, smplx_verts_tensor.T).T 
rot_human_verts_tensor = torch.matmul(std2mesh, human_verts_tensor.T).T 

h2s_map_dist, h2s_map_face, h2s_map_uvw = smplx_mesh_BVH.signed_distance(human_verts_tensor, return_uvw=True, mode="raystab")
human_lbs_weights = (smplx_lbs_weights[smplx_faces_tensor[h2s_map_face], :] * h2s_map_uvw[:, :, None]).sum(1)

body_pose = init_body_pose
smplx_output = smplx_model(betas=betas.unsqueeze(0), body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0), return_verts=True)
original_smplx_verts_tensor = smplx_output.vertices[0].detach()

for idx in range(1):
    os.makedirs(f"{save_path}/{test_name}_random/render_normal", exist_ok=True)
    os.makedirs(f"{save_path}/{test_name}_random/render_rgb", exist_ok=True)

    for i in tqdm(range(120)):

        new_betas = copy.deepcopy(betas) * (1- i/120) + torch.tensor([-3.9661, -3.0112,  2.1967,  0.9014,  1.2515,  1.8726, -4.7323,  4.9004, -4.7780,  3.0193]) * (i/120) # Change the shape parameters of SMPL-X

        smplx_output = smplx_model(betas=new_betas.unsqueeze(0), body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0), return_verts=True)
        new_smplx_verts_tensor = smplx_output.vertices[0].detach()
        new_offset = (new_smplx_verts_tensor - original_smplx_verts_tensor).to(device)

        human_offset = (new_offset[smplx_faces_tensor[h2s_map_face], :] * h2s_map_uvw[:, :, None]).sum(1)
        new_human_verts_tensor = rot_human_verts_tensor + human_offset

        scaled_human_verts_tensor = torch.matmul(mesh2std, new_human_verts_tensor.T).T 
        scaled_human_verts_tensor = (scaled_human_verts_tensor - center) / scale * 0.7
        mesh = Mesh(v_pos=scaled_human_verts_tensor.contiguous(), t_pos_idx=human_faces_tensor.contiguous())
        render_out = render_mesh(ctx=ctx, mesh=mesh,
            mvp_mtx=data["mvp_mtx"],
            c2w=data["c2w"],
            camera_positions=data["camera_positions"],
            light_positions=data["light_positions"],
            height=data["height"],
            width=data["width"],
            render_rgb=True,
            mesh_rgb=human_rgb_tensor
            )
        normal_img = ((render_out["normal"][0] + (1 - render_out["mask"][0])) * 255).detach().cpu().numpy().astype(np.uint8)
        rgb_img = ((render_out["rgb"][0]) * 255).detach().cpu().numpy().astype(np.uint8)
        name = str(i).rjust(6,"0")
        Image.fromarray(normal_img).save(f"{save_path}/{test_name}_random/render_normal/{name}.jpg")
        Image.fromarray(rgb_img).save(f"{save_path}/{test_name}_random/render_rgb/{name}.jpg")

        # break

    normal_img_files = sorted(glob(os.path.join(f"{save_path}/{test_name}_random/render_normal", '*.jpg')))
    rgb_img_files = sorted(glob(os.path.join(f"{save_path}/{test_name}_random/render_rgb", '*.jpg')))

    normal_imgs = [cv2.imread(f) for f in normal_img_files]
    normal_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in normal_imgs]
    imageio.mimsave(f"{save_path}/{test_name}_random/normal_render.mp4", normal_imgs, fps=30)

    rgb_imgs = [cv2.imread(f) for f in rgb_img_files]
    rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_imgs]
    imageio.mimsave(f"{save_path}/{test_name}_random/rgb_render.mp4", rgb_imgs, fps=30)