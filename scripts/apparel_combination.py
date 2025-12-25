import os
import sys
import imageio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import trimesh
from glob import glob
import cubvh
import numpy as np
from threestudio.utils.rasterize import NVDiffRasterizerContext
import torch
import cv2
from PIL import Image
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *
from threestudio.utils.rendering import *
import heapq
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

naked_human_dir_path = ["./outputs/naked_human/stage1-geometry/A_strong_grown_man_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_strong_grown_man_wearing_cargo_pants_brown_leather_boots_a_red_plaid_flannel_shirt_a_work_glove_on_his_left_hand_and_a_silver_wristwatch_on_his_right_hand/save/obj/result/comp_0.obj"] # Replace this with the path to your generated naked human
cloth_dir_path_list = [
    ["./outputs/naked_human/stage1-geometry/A_thin_boy_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_thin_boy_wearing_blue_denim_shorts_white_high-top_sneakers_a_cotton_t-shirt_a_navy_baseball_cap_and_a_sports_watch_on_his_left_hand/save/obj/result/comp_1.obj"],
    ["./outputs/naked_human/stage1-geometry/A_thin_boy_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_thin_boy_wearing_blue_denim_shorts_white_high-top_sneakers_a_cotton_t-shirt_a_navy_baseball_cap_and_a_sports_watch_on_his_left_hand/save/obj/result/comp_2.obj"],
    ["./outputs/naked_human/stage1-geometry/A_thin_boy_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_thin_boy_wearing_blue_denim_shorts_white_high-top_sneakers_a_cotton_t-shirt_a_navy_baseball_cap_and_a_sports_watch_on_his_left_hand/save/obj/result/comp_3.obj"],
    ["./outputs/naked_human/stage1-geometry/A_thin_boy_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_thin_boy_wearing_blue_denim_shorts_white_high-top_sneakers_a_cotton_t-shirt_a_navy_baseball_cap_and_a_sports_watch_on_his_left_hand/save/obj/result/comp_4.obj"],
    ["./outputs/naked_human/stage1-geometry/A_thin_boy_in_his_underwear", "./outputs/clothed_human/stage3-fine-texture/A_thin_boy_wearing_blue_denim_shorts_white_high-top_sneakers_a_cotton_t-shirt_a_navy_baseball_cap_and_a_sports_watch_on_his_left_hand/save/obj/result/comp_5.obj"],
] # Replace this with the path to your generated clothing

device = "cuda"
save_path = "./outputs/transfer_cloth"
os.makedirs(os.path.join(save_path, "render_normal"), exist_ok=True)
os.makedirs(os.path.join(save_path, "render_rgb"), exist_ok=True)

target_cloth_mesh_list = []

for idx in range(len(cloth_dir_path_list)):
    source_naked_human_dir_path = cloth_dir_path_list[idx][0]
    source_cloth_dir_path = cloth_dir_path_list[idx][1]
    target_naked_human_dir_path = naked_human_dir_path[0]

    target_human_mesh = trimesh.load(naked_human_dir_path[1])
    source_cloth_mesh = trimesh.load(source_cloth_dir_path)

    components = source_cloth_mesh.split(only_watertight=False)
    if os.path.basename(source_cloth_dir_path) == "comp_2.obj":
        largest_components = heapq.nlargest(2, components, key=lambda comp: len(comp.faces))
    else:
        largest_components = heapq.nlargest(1, components, key=lambda comp: len(comp.faces))
    source_cloth_mesh = trimesh.util.concatenate(largest_components)

    source_naked_smplx_mesh = trimesh.load(os.path.join(source_naked_human_dir_path, "save/cache/obj/smplx/temp_5005.obj"))
    target_naked_smplx_mesh = trimesh.load(os.path.join(target_naked_human_dir_path, "save/cache/obj/smplx/temp_5005.obj"))
    source_naked_human_mesh = trimesh.load(os.path.join(source_naked_human_dir_path, "save/obj/result/human.obj"))
    target_naked_human_mesh = trimesh.load(os.path.join(target_naked_human_dir_path, "save/obj/result/human.obj"))

    source_naked_smplx_mesh_BVH = cubvh.cuBVH(source_naked_smplx_mesh.vertices, source_naked_smplx_mesh.faces)
    target_naked_smplx_mesh_BVH = cubvh.cuBVH(target_naked_smplx_mesh.vertices, target_naked_smplx_mesh.faces)
    source_naked_human_mesh_BVH = cubvh.cuBVH(source_naked_human_mesh.vertices, source_naked_human_mesh.faces)
    target_naked_human_mesh_BVH = cubvh.cuBVH(target_naked_human_mesh.vertices, target_naked_human_mesh.faces)

    source_smplx_verts_tensor = torch.from_numpy(source_naked_smplx_mesh.vertices).to(device).float()
    source_human_verts_tensor = torch.from_numpy(source_naked_human_mesh.vertices).to(device).float()
    source_cloth_verts_tensor = torch.from_numpy(source_cloth_mesh.vertices).to(device).float()
    source_smplx_faces_tensor = torch.from_numpy(source_naked_smplx_mesh.faces).to(device).long()
    source_human_faces_tensor = torch.from_numpy(source_naked_human_mesh.faces).to(device).long()
    source_cloth_faces_tensor = torch.from_numpy(source_cloth_mesh.faces).to(device).long()

    target_smplx_verts_tensor = torch.from_numpy(target_naked_smplx_mesh.vertices).to(device).float()
    target_human_verts_tensor = torch.from_numpy(target_naked_human_mesh.vertices).to(device).float()
    target_smplx_faces_tensor = torch.from_numpy(target_naked_smplx_mesh.faces).to(device).long()
    target_human_faces_tensor = torch.from_numpy(target_naked_human_mesh.faces).to(device).long()

    source_c2h_map_dist, source_c2h_map_face, source_c2h_map_uvw = source_naked_human_mesh_BVH.signed_distance(source_cloth_verts_tensor, return_uvw=True, mode="raystab")
    source_c2h_map_vert = (source_human_verts_tensor[source_human_faces_tensor[source_c2h_map_face], :] * source_c2h_map_uvw[:, :, None]).sum(1)

    source_h2s_map_dist, source_h2s_map_face, source_h2s_map_uvw = source_naked_smplx_mesh_BVH.signed_distance(source_c2h_map_vert, return_uvw=True, mode="raystab")
    source_h2s_map_vert = (source_smplx_verts_tensor[source_smplx_faces_tensor[source_h2s_map_face], :] * source_h2s_map_uvw[:, :, None]).sum(1)

    target_h2s_map_vert = (target_smplx_verts_tensor[target_smplx_faces_tensor[source_h2s_map_face], :] * source_h2s_map_uvw[:, :, None]).sum(1)

    target_s2h_map_dist, target_s2h_map_face, target_s2h_map_uvw = target_naked_human_mesh_BVH.signed_distance(target_h2s_map_vert, return_uvw=True, mode="raystab")
    target_s2h_map_vert = (target_human_verts_tensor[target_human_faces_tensor[target_s2h_map_face], :] * target_s2h_map_uvw[:, :, None]).sum(1)

    target_cloth_verts_tensor = target_s2h_map_vert - source_c2h_map_vert + source_cloth_verts_tensor

    source_cloth_mesh.vertices = target_cloth_verts_tensor.cpu().detach().numpy()
    target_cloth_mesh_list.append(source_cloth_mesh)

ctx = NVDiffRasterizerContext("cuda", device)
render_data = prepare_render_data(120, device)

with torch.no_grad():
    for i in range(0, 120, 1):
        data = sample_render_data(render_data, i)

        human_mesh = trimesh.util.concatenate(target_cloth_mesh_list + [target_human_mesh])

        human_verts_tensor = torch.from_numpy(human_mesh.vertices).to(device).float()
        human_faces_tensor = torch.from_numpy(human_mesh.faces).to(device).long()
        human_rgb_tensor = (torch.from_numpy(human_mesh.visual.vertex_colors) / 255).float().to(device)[:, :3]
        mesh = Mesh(v_pos=human_verts_tensor.contiguous(), t_pos_idx=human_faces_tensor.contiguous())
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
        Image.fromarray(normal_img).save(f"{save_path}/render_normal/{name}.jpg")
        Image.fromarray(rgb_img).save(f"{save_path}/render_rgb/{name}.jpg")

    normal_img_files = sorted(glob(os.path.join(f"{save_path}/render_normal", '*.jpg')))
    rgb_img_files = sorted(glob(os.path.join(f"{save_path}/render_rgb", '*.jpg')))

    normal_imgs = [cv2.imread(f) for f in normal_img_files]
    normal_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in normal_imgs]
    imageio.mimsave(f"{save_path}/normal_render.mp4", normal_imgs, fps=30)

    rgb_imgs = [cv2.imread(f) for f in rgb_img_files]
    rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_imgs]
    imageio.mimsave(f"{save_path}/rgb_render.mp4", rgb_imgs, fps=30)






