import os 
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dict_path', default="./load/barbie/data_dict.json", type=str)
parser.add_argument('--naked_human_exp_root_dir', default="./outputs/naked_human", type=str)
parser.add_argument('--naked_human_idx', default="0:1:1", type=str)
parser.add_argument('--gpu_idx', default=0, type=int)
args = parser.parse_args()

with open(args.dict_path, 'r') as f:
    dict_data = json.load(f)

naked_human_exp_root_dir=args.naked_human_exp_root_dir
exp_name1="stage1-geometry"
exp_name2="stage2-coarse-texture"
exp_name3="stage3-fine-texture"
isosurface_resolution = "shell:0.05:0.9:5e-8"
gpu_idx=args.gpu_idx
naked_human_idx_list = args.naked_human_idx.split(":")
naked_human_idx_list = list(range(int(naked_human_idx_list[0]), int(naked_human_idx_list[1]), int(naked_human_idx_list[2])))

for i in range(len(naked_human_idx_list)):
    naked_human_prompt = dict_data["naked_human_prompts_list"][naked_human_idx_list[i]]
    naked_human_tag = naked_human_prompt.replace(" ", "_").replace(".", "").replace(",", "")
    test_save_path=f"{naked_human_exp_root_dir}/{exp_name1}/{naked_human_tag}/save/cache"
    geometry_convert_from=f"{naked_human_exp_root_dir}/{exp_name1}/{naked_human_tag}/ckpts/last.ckpt" 
    resume=f"{naked_human_exp_root_dir}/{exp_name2}/{naked_human_tag}/ckpts/last.ckpt" 

    command1 = ["python", "launch.py", "--config", "configs/human-geometry.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={naked_human_tag}", f"name={exp_name1}", f"exp_root_dir={naked_human_exp_root_dir}",
                "data.sampling_type=full_body", f"system.geometry.isosurface_resolution={isosurface_resolution}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={naked_human_prompt}, black background, normal map", 
                f"system.prompt_processor_add.prompt={naked_human_prompt}, black background, depth map", 
                "system.prompt_processor.human_part_prompt=false", "system.geometry.use_sdf_loss=false", 
                "system.geometry.shape_init=opt_smplx:dapose1:10:3:True", "system.loss.lambda_smplx_local=1000",
                "system.loss.lambda_smplx_global=1000", "system.geometry.update_smplx_loss_step=1001", "system.exporter.save_flag=true"]

    command2 = ["python", "launch.py", "--config", "configs/human-texture-coarse.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={naked_human_tag}", f"name={exp_name2}", f"exp_root_dir={naked_human_exp_root_dir}",
                "data.sampling_type=full_body", f"system.geometry_convert_from={geometry_convert_from}", f"system.geometry.isosurface_resolution={isosurface_resolution}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={naked_human_prompt}", 
                "system.prompt_processor.human_part_prompt=false", "system.exporter.save_flag=true"]

    command3 = ["python", "launch.py", "--config", "configs/human-texture-fine.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={naked_human_tag}", f"name={exp_name3}", f"exp_root_dir={naked_human_exp_root_dir}",
                f"system.geometry_convert_from={geometry_convert_from}", f"system.geometry.isosurface_resolution={isosurface_resolution}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={naked_human_prompt}", 
                f"data.dataroot={test_save_path}", f"resume={resume}",
                "system.prompt_processor.human_part_prompt=false", "system.exporter.save_flag=true"]

    tmp_neg_prompt_flag = False
    tmp_neg_prompt = ""
    if "strong" not in naked_human_prompt:
        tmp_neg_prompt += "too strong body"
        tmp_neg_prompt_flag = True
    if "old" in naked_human_prompt or "middle-aged" in naked_human_prompt or "mature" in naked_human_prompt or "grandmother" in naked_human_prompt or "grandfather" in naked_human_prompt:
        if tmp_neg_prompt == "":
            tmp_neg_prompt += "glasses"
        else:
            tmp_neg_prompt += ", glasses"
        tmp_neg_prompt_flag = True

    if tmp_neg_prompt_flag:
        command1.append(f"system.prompt_processor.negative_prompt={tmp_neg_prompt}")
        command1.append(f"system.prompt_processor_add.negative_prompt={tmp_neg_prompt}")

    command1.append(f"trainer.max_steps=6000")
    command2.append(f"trainer.max_steps=2000")
    command3.append(f"trainer.max_steps=10000")

    tmp_str = ""
    for c in command1:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{naked_human_exp_root_dir}/{exp_name1}/{naked_human_tag}"])
    subprocess.run(command1)

    tmp_str = ""
    for c in command2:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{naked_human_exp_root_dir}/{exp_name2}/{naked_human_tag}"])
    subprocess.run(command2)

    tmp_str = ""
    for c in command3:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{naked_human_exp_root_dir}/{exp_name3}/{naked_human_tag}"])
    subprocess.run(command3)