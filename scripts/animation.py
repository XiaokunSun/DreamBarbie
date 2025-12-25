
import os
import torch
import trimesh
from glob import glob
import cubvh
import numpy as np
import sys
import imageio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threestudio.utils import smplx_customized
import pickle
import torch
import heapq
from tqdm import tqdm
import cv2
from PIL import Image
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.utils.rendering import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

human_mesh_dir_path = "./outputs/clothed_human/stage3-fine-texture/A_strong_grown_man_wearing_cargo_pants_brown_leather_boots_a_red_plaid_flannel_shirt_a_work_glove_on_his_left_hand_and_a_silver_wristwatch_on_his_right_hand" # Replace this with the path to your generated clothed human
smplx_mesh_dir_path = "./outputs/naked_human/stage1-geometry/A_strong_grown_man_in_his_underwear" # Replace this with the path to your generated naked human
save_path = "./outputs/animation"
device = "cuda"
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
v_offsets = human_ckpt["state_dict"]["geometry.betas"].cpu()

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

with open('./load/smplx_models/smplx/smplx_watertight.pkl', 'rb') as f:
    tmp_data = pickle.load(f, encoding='latin1')
smplx_model.faces = tmp_data["smpl_F"].numpy()
smplx_lbs_weights = smplx_model.lbs_weights.to(device).float()
init_body_pose = np.zeros((21, 3), dtype=np.float32)
init_body_pose[0, 1] = 0.2
init_body_pose[0, 2] = 0.2
init_body_pose[1, 1] = -0.2
init_body_pose[1, 2] = -0.2
init_body_pose[15, 2] = -0.7
init_body_pose[16, 2] = 0.7
init_jaw_pose = np.array([0.1, 0, 0], dtype=np.float32)
smplx_mesh = trimesh.load(os.path.join(smplx_mesh_dir_path, "save/cache/obj/smplx/temp_5005.obj"))
human_mesh_list = []
for mesh_path in sorted(glob(os.path.join(human_mesh_dir_path, "save/obj/result/comp_*.obj"))):
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

rot_smplx_verts_tensor = torch.matmul(std2mesh, smplx_verts_tensor.T).T 
rot_human_verts_tensor = torch.matmul(std2mesh, human_verts_tensor.T).T 

rot_smplx_verts_tensor = torch.cat((rot_smplx_verts_tensor, torch.zeros((10475 - 9383, 3)).to(device).float()))
h2s_map_dist, h2s_map_face, h2s_map_uvw = smplx_mesh_BVH.signed_distance(human_verts_tensor, return_uvw=True, mode="raystab")
human_lbs_weights = (smplx_lbs_weights[smplx_faces_tensor[h2s_map_face], :] * h2s_map_uvw[:, :, None]).sum(1)

ctx = NVDiffRasterizerContext("cuda", device)
render_data = prepare_render_data(120, device)
data = sample_render_data(render_data, 0)

with torch.no_grad():
    motion_seq_list = []
    trans_seq_list = []
    name_seq_list = []
    path_list = sorted(glob("/root/data/SunXiaoKun/Data/AIST++/*.pkl"))[0:1] # Replace this with the path to your AIST++ dataset (https://google.github.io/aistplusplus_dataset)
    for path in path_list:
        with open(path, 'rb') as f:
            temp_data = pickle.load(f)
        trans = temp_data["smpl_trans"] / temp_data["smpl_scaling"]
        trans = trans - trans.mean(0)
        trans_seq_list.append(trans)
        motions = temp_data["smpl_poses"].reshape(-1, 24, 3)[:, 1:22, :]
        motion_seq_list.append(motions)
        name_seq_list.append(path.split("/")[-1].replace(".pkl", ""))

    for idx, motion_seq in enumerate(motion_seq_list):
        seq_name = name_seq_list[idx]
        trans_seq = trans_seq_list[idx]
        os.makedirs(f"{save_path}/{seq_name}/render_normal", exist_ok=True)
        os.makedirs(f"{save_path}/{seq_name}/render_rgb", exist_ok=True)
        print(seq_name)
        print(motion_seq.shape[0])
        for i in tqdm(range(0, motion_seq.shape[0])):

            smplx_output = smplx_model(betas=betas.unsqueeze(0), body_pose=torch.tensor(-init_body_pose, dtype=torch.float32).unsqueeze(0), jaw_pose=torch.tensor(-init_jaw_pose, dtype=torch.float32).unsqueeze(0), return_verts=True, vertices=rot_smplx_verts_tensor[None].cpu())
            a2l_def_mats = smplx_output.A_processed.detach().to(device)
            rigid_def_mats = torch.einsum('bnj,bjxy->bnxy', human_lbs_weights[None], a2l_def_mats)
            animated_human_verts_tensor = (torch.einsum('bnxy,bny->bnx', rigid_def_mats[..., :3, :3], rot_human_verts_tensor[None]) + rigid_def_mats[..., :3, 3])[0]

            rigid_def_mats = torch.einsum('bnj,bjxy->bnxy', smplx_lbs_weights[None], a2l_def_mats)
            animated_smplx_verts_tensor = (torch.einsum('bnxy,bny->bnx', rigid_def_mats[..., :3, :3], rot_smplx_verts_tensor[None]) + rigid_def_mats[..., :3, 3])[0]

            smplx_output = smplx_model(betas=betas.unsqueeze(0), body_pose=torch.tensor(motion_seq[i, ...], dtype=torch.float32).unsqueeze(0), return_verts=True, vertices=animated_smplx_verts_tensor[None].cpu())
            a2l_def_mats = smplx_output.A_processed.detach().to(device)
            rigid_def_mats = torch.einsum('bnj,bjxy->bnxy', human_lbs_weights[None], a2l_def_mats)
            animated_human_verts_tensor = (torch.einsum('bnxy,bny->bnx', rigid_def_mats[..., :3, :3], animated_human_verts_tensor[None]) + rigid_def_mats[..., :3, 3])[0]

            animated_human_verts_tensor += torch.tensor(trans_seq[i], dtype=torch.float32).to(device)

            animated_human_verts_tensor = torch.matmul(mesh2std, animated_human_verts_tensor.T).T 
            mesh = Mesh(v_pos=animated_human_verts_tensor.contiguous(), t_pos_idx=human_faces_tensor.contiguous())

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
            Image.fromarray(normal_img).save(f"{save_path}/{seq_name}/render_normal/{name}.jpg")
            Image.fromarray(rgb_img).save(f"{save_path}/{seq_name}/render_rgb/{name}.jpg")
            # break

        normal_img_files = sorted(glob(os.path.join(f"{save_path}/{seq_name}/render_normal", '*.jpg')))
        rgb_img_files = sorted(glob(os.path.join(f"{save_path}/{seq_name}/render_rgb", '*.jpg')))

        normal_imgs = [cv2.imread(f) for f in normal_img_files]
        normal_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in normal_imgs]
        imageio.mimsave(f"{save_path}/{seq_name}/normal_render.mp4", normal_imgs, fps=45)

        rgb_imgs = [cv2.imread(f) for f in rgb_img_files]
        rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_imgs]
        imageio.mimsave(f"{save_path}/{seq_name}/rgb_render.mp4", rgb_imgs, fps=45)