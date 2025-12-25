import smplx
import trimesh
import torch
import numpy as np
from pysdf import SDF
import cubvh
import os
import pickle

vis_dir_path = "./outputs/smplx_cloth_mask"
os.makedirs(vis_dir_path, exist_ok=True)
smplx_model = smplx.create(
    "./load/smplx_models",
    model_type="smplx",
    gender="neutral",
    use_face_contour=True,
    num_betas=10,
    num_expression_coeffs=10,
    ext="npz",
    use_pca=False,  # explicitly control hand pose
    flat_hand_mean=True,  # use a flatten hand default pose
)

body_pose = np.zeros((21, 3), dtype=np.float32)

smplx_output = smplx_model(body_pose=torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0), return_verts=True)
smplx_vertices = smplx_output.vertices.detach().cpu().numpy()[0]
smplx_faces = smplx_model.faces

smplx_mesh = trimesh.Trimesh(
    vertices=smplx_vertices,
    faces=smplx_faces,
)

smplx_lbs_weights = smplx_model.lbs_weights.cpu().numpy()
smplx_vertices = smplx_output.vertices.detach().cpu().numpy()[0]
smplx_faces = smplx_model.faces
trimesh.Trimesh(vertices=smplx_vertices, faces=smplx_faces).export(f"{vis_dir_path}/smplx_mesh.obj")
cloth_mask_dict = {}
LBS_argmax = np.argmax(smplx_lbs_weights, axis=-1)

part_mask = np.zeros_like(LBS_argmax)
name = "scarf"
for part_idx in [12]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_glove"
for part_idx in [21] + list(range(40, 55)):
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_glove"
for part_idx in [20] + list(range(25, 40)):
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_glove"
for part_idx in [21] + list(range(40, 55)):
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "gloves"
part_mask = np.logical_or(cloth_mask_dict["left_glove"], cloth_mask_dict["right_glove"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_shoe"
for part_idx in [7, 10]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_shoe"
for part_idx in [8, 11]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "shoes"
part_mask = np.logical_or(cloth_mask_dict["left_shoe"], cloth_mask_dict["right_shoe"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "long_shoes"
part_mask = (smplx_vertices[:, 1] < ((smplx_output.joints[0, 5].detach().cpu().numpy() * 0.5 + smplx_output.joints[0, 8].detach().cpu().numpy() * 0.5))[1])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "super_long_shoes"
part_mask = (smplx_vertices[:, 1] < ((smplx_output.joints[0, 5].detach().cpu().numpy() * 1.0 + smplx_output.joints[0, 8].detach().cpu().numpy() * 0.0))[1])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "short_bottom"
for part_idx in [0, 1, 2]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] < (smplx_output.joints[0, 0, 1].detach().cpu().numpy() + 0.05))
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "super_short_bottom"
part_mask = np.logical_and(cloth_mask_dict["short_bottom"], smplx_vertices[:, 1] > ((smplx_output.joints[0, 1].detach().cpu().numpy() + smplx_output.joints[0, 4].detach().cpu().numpy()) / 2)[1])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "long_bottom"
for part_idx in [4, 5]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_or(part_mask, cloth_mask_dict["short_bottom"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "skirt"
part_mask = np.logical_and(cloth_mask_dict["short_bottom"], smplx_vertices[:, 1] > ((smplx_output.joints[0, 1].detach().cpu().numpy() * 0.98 + smplx_output.joints[0, 4].detach().cpu().numpy() * 0.02))[1])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
for part_idx in [9, 13, 14]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
N = 100
arr = smplx_lbs_weights[part_mask, 12]
indices = np.argpartition(arr, -N)[-N:]
indices = indices[np.argsort(-arr[indices])]
tmp_mask = np.zeros_like(part_mask)
tmp_mask[np.where(part_mask == True)[0][indices]] = True
tmp_part_mask = ~np.logical_and(part_mask, tmp_mask)
part_mask = np.zeros_like(LBS_argmax)
name = "tank_top"
for part_idx in [0, 3, 6, 9, 13, 14]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] > (smplx_output.joints[0, 0, 1].detach().cpu().numpy() - 0.05))
part_mask = np.logical_and(tmp_part_mask, part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "super_short_top"
for part_idx in [0, 3, 6, 9, 13, 14]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] > (smplx_output.joints[0, 0, 1].detach().cpu().numpy() - 0.05))
part_mask[[5975, 5976, 6705, 6692, 6704, 6667, 6666, 6698, 6697, 6218, 6219, 6220, 3353, 3459, 3458, 3949, 3950, 3918, 3919, 3956, 3957, 3213, 3212]] = True
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "short_top"
for part_idx in [16, 17]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_or(part_mask, cloth_mask_dict["super_short_top"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "long_top"
for part_idx in [18, 19]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_or(part_mask, cloth_mask_dict["short_top"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_watch"
part_mask = np.logical_and(smplx_vertices[:, 0] >= (0.18 * smplx_output.joints[0, 18].detach().cpu().numpy() + 0.82 * smplx_output.joints[0, 20].detach().cpu().numpy())[0], cloth_mask_dict["long_top"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_watch"
part_mask = np.logical_and(smplx_vertices[:, 0] <= (0.18 * smplx_output.joints[0, 19].detach().cpu().numpy() + 0.82 * smplx_output.joints[0, 21].detach().cpu().numpy())[0], cloth_mask_dict["long_top"])
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "face_mask"
for part_idx in [15, 22]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 2] > smplx_output.joints[0, 127].detach().cpu().numpy()[2], part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 135].detach().cpu().numpy()[1], part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] < smplx_output.joints[0, 88].detach().cpu().numpy()[1] - 0.01, part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "glasses"
for part_idx in [15, 22]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 2] > smplx_output.joints[0, 58].detach().cpu().numpy()[2] - 0.02, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] < smplx_output.joints[0, 80].detach().cpu().numpy()[1] + 0.01, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 88].detach().cpu().numpy()[1] - 0.01, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 2] < smplx_output.joints[0, 86].detach().cpu().numpy()[2], part_mask)
a = np.logical_and(smplx_vertices[:, 2] >= 0.0489691, part_mask)
b = np.logical_and(np.logical_and(smplx_vertices[:, 2] < 0.0489691, smplx_vertices[:, 1] > (0.304542 - 0.005)), np.logical_and(part_mask, smplx_vertices[:, 1] < (0.304542 + 0.025)))
part_mask = np.logical_or(a, b)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "headset"
for part_idx in [15, 22]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 2] < smplx_output.joints[0, 127].detach().cpu().numpy()[2], part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 130].detach().cpu().numpy()[1], part_mask)
part_mask = np.logical_and(smplx_vertices[:, 2] > (smplx_output.joints[0, 127].detach().cpu().numpy()[2] - 0.045), part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "hairband"
for part_idx in [15, 22]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 2] < smplx_output.joints[0, 127].detach().cpu().numpy()[2] - 0.012, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 130].detach().cpu().numpy()[1] + 0.065, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 2] > (smplx_output.joints[0, 127].detach().cpu().numpy()[2] - 0.048), part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_knee_guard"
for part_idx in [1, 4]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] < (smplx_output.joints[0, 1, 1].detach().cpu().numpy() * 0.3 + smplx_output.joints[0, 4, 1].detach().cpu().numpy() * 0.7))
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] > (smplx_output.joints[0, 4, 1].detach().cpu().numpy() * 0.7 + smplx_output.joints[0, 7, 1].detach().cpu().numpy() * 0.3))
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_knee_guard"
for part_idx in [2, 5]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] < (smplx_output.joints[0, 2, 1].detach().cpu().numpy() * 0.3 + smplx_output.joints[0, 5, 1].detach().cpu().numpy() * 0.7))
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] > (smplx_output.joints[0, 5, 1].detach().cpu().numpy() * 0.7 + smplx_output.joints[0, 8, 1].detach().cpu().numpy() * 0.3))
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_arm_guard"
for part_idx in [16, 18]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 0] > (smplx_output.joints[0, 16, 0].detach().cpu().numpy() * 0.3 + smplx_output.joints[0, 18, 0].detach().cpu().numpy() * 0.7))
part_mask = np.logical_and(part_mask, smplx_vertices[:, 0] < (smplx_output.joints[0, 18, 0].detach().cpu().numpy() * 0.7 + smplx_output.joints[0, 20, 0].detach().cpu().numpy() * 0.3))
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_arm_guard"
for part_idx in [17, 19]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 0] < (smplx_output.joints[0, 17, 0].detach().cpu().numpy() * 0.3 + smplx_output.joints[0, 19, 0].detach().cpu().numpy() * 0.7))
part_mask = np.logical_and(part_mask, smplx_vertices[:, 0] > (smplx_output.joints[0, 19, 0].detach().cpu().numpy() * 0.7 + smplx_output.joints[0, 21, 0].detach().cpu().numpy() * 0.3))
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "necklace"
for part_idx in [9, 13, 14]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
N = 70
arr = smplx_lbs_weights[part_mask, 12]
indices = np.argpartition(arr, -N)[-N:]
indices = indices[np.argsort(-arr[indices])]
tmp_mask = np.zeros_like(part_mask)
tmp_mask[np.where(part_mask == True)[0][indices]] = True
part_mask = np.logical_and(part_mask, tmp_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "cap"
for part_idx in [15, 22, 23, 24]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 80].detach().cpu().numpy()[1] + 0.01, part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "headband"
for part_idx in [15, 22, 23, 24]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(smplx_vertices[:, 1] < smplx_output.joints[0, 80].detach().cpu().numpy()[1] + 0.04, part_mask)
part_mask = np.logical_and(smplx_vertices[:, 1] > smplx_output.joints[0, 80].detach().cpu().numpy()[1] + 0.005, part_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "left_ring"
for part_idx in [25]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
arr = smplx_lbs_weights[part_mask, 25]
N = 20
indices = np.argpartition(arr, -N)[-N:]
indices = indices[np.argsort(-arr[indices])]
indices = np.delete(indices, 14)
tmp_mask = np.zeros_like(part_mask)
tmp_mask[np.where(part_mask == True)[0][indices]] = True
part_mask = np.logical_and(part_mask, tmp_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "right_ring"
for part_idx in [40]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
arr = smplx_lbs_weights[part_mask, 40]
N = 20
indices = np.argpartition(arr, -N)[-N:]
indices = indices[np.argsort(-arr[indices])]
indices = np.delete(indices, 14)
tmp_mask = np.zeros_like(part_mask)
tmp_mask[np.where(part_mask == True)[0][indices]] = True
part_mask = np.logical_and(part_mask, tmp_mask)
cloth_mask_dict[name] = part_mask

part_mask = np.zeros_like(LBS_argmax)
name = "hot_top"
for part_idx in [3, 6, 9]:
    part_mask = np.logical_or(part_mask, LBS_argmax == part_idx)
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] < (smplx_output.joints[0, 9, 1].detach().cpu().numpy() * 0.5 + smplx_output.joints[0, 12, 1].detach().cpu().numpy() * 0.5))
part_mask = np.logical_and(part_mask, smplx_vertices[:, 1] > smplx_output.joints[0, 3, 1].detach().cpu().numpy() + 0.1)
cloth_mask_dict[name] = part_mask

print(cloth_mask_dict.keys())

for cloth_name, cloth_mask in cloth_mask_dict.items():
    inside_faces = [f for f in smplx_faces if np.all(cloth_mask[f])]
    used_vertices = np.unique(np.hstack(inside_faces))
    new_vertices = smplx_vertices[used_vertices]
    vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
    new_mesh.export(f"{vis_dir_path}/{cloth_name}_mesh.obj")

with open(f'./load/smplx_models/smplx/smplx_cloth_mask.pkl', 'wb') as f:
    pickle.dump(cloth_mask_dict, f)