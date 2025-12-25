import os 
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dict_path', default="./load/barbie/data_dict.json", type=str)
parser.add_argument('--naked_human_exp_root_dir', default="./outputs/naked_human", type=str)
parser.add_argument('--clothed_human_exp_root_dir', default="./outputs/clothed_human", type=str)
parser.add_argument('--naked_human_idx', default="0:1:1", type=str)
parser.add_argument('--cloth_idx', default="0:1:1", type=str)
parser.add_argument('--gpu_idx', default=0, type=int)
args = parser.parse_args()

with open(args.dict_path, 'r') as f:
    dict_data = json.load(f)

naked_human_exp_root_dir=args.naked_human_exp_root_dir
clothed_human_exp_root_dir=args.clothed_human_exp_root_dir
exp_name1="stage1-geometry"
exp_name2="stage2-coarse-texture"
exp_name3="stage3-fine-texture"
gpu_idx=args.gpu_idx

naked_human_idx_list = args.naked_human_idx.split(":")
cloth_idx_list = args.cloth_idx.split(":")
naked_human_idx_list = list(range(int(naked_human_idx_list[0]), int(naked_human_idx_list[1]), int(naked_human_idx_list[2])))
cloth_idx_list = list(range(int(cloth_idx_list[0]), int(cloth_idx_list[1]), int(cloth_idx_list[2])))

assert len(naked_human_idx_list) == len(cloth_idx_list)

for i in range(len(naked_human_idx_list)):
    naked_human_prompt = dict_data["naked_human_prompts_list"][naked_human_idx_list[i]]
    cloth_prompts_list = dict_data["cloth_prompts_list"][cloth_idx_list[i]]
    name_list = dict_data["name_list"][cloth_idx_list[i]]
    HN_naked_human_prompt = " ".join(naked_human_prompt.split(" ")[:-3])
    gender = naked_human_prompt.split(" ")[-2]
    HN_cloth_prompts_list = []
    for cloth_idx in range(len(cloth_prompts_list)):
        if name_list[cloth_idx + 1] == "left_watch":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} left hand")
        elif name_list[cloth_idx + 1] == "right_watch":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} right hand")
        elif name_list[cloth_idx + 1] == "left_glove":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} left hand")
        elif name_list[cloth_idx + 1] == "right_glove":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} right hand")
        elif name_list[cloth_idx + 1] == "left_arm_guard":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} left arm")
        elif name_list[cloth_idx + 1] == "right_arm_guard":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} right arm")
        elif name_list[cloth_idx + 1] == "left_knee_guard":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} left knee")
        elif name_list[cloth_idx + 1] == "right_knee_guard":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} right knee")
        elif name_list[cloth_idx + 1] == "left_shoe":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} left foot")
        elif name_list[cloth_idx + 1] == "right_shoe":
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", "") + f" on {gender} right foot")
        else:
            HN_cloth_prompts_list.append(cloth_prompts_list[cloth_idx].lower().replace("a pair of ", "").replace(".", ""))
    data_part_ratio_list=[]
    for cloth_idx in range(len(cloth_prompts_list)):
        if name_list[cloth_idx + 1] in ["cap", "hairband", "headset", "headband", "glasses", "face_mask"]:
            data_part_ratio_list.append([[0.0,0.7,0.3,0.0]])
        elif name_list[cloth_idx + 1] in ["long_top", "short_top", "super_short_top", "tank_top", "hot_top", "left_glove", "right_glove", "gloves", "left_watch", "right_watch", "necklace", "scarf", "left_arm_guard", "right_arm_guard"]:
            data_part_ratio_list.append([[0.3,0.0,0.7,0.0]])
        elif name_list[cloth_idx + 1] in ["long_bottom", "short_bottom", "super_short_bottom", "long_skirt", "short_skirt", "super_short_skirt", "shoes", "left_shoe", "right_shoe", "long_shoes", "super_long_shoes", "left_knee_guard", "right_knee_guard"]:
            data_part_ratio_list.append([[0.3,0.0,0.0,0.7]])
        else:
            assert False

    HN_prompts_list = [
        f"{HN_naked_human_prompt} with no shoes, no socks, no shirt, no tops, just {HN_cloth_prompts_list[0]}",
        f"{HN_naked_human_prompt} with no shirt, no tops, just {HN_cloth_prompts_list[0]} and {HN_cloth_prompts_list[1]}",
        f"{HN_naked_human_prompt} wearing {HN_cloth_prompts_list[0]}, {HN_cloth_prompts_list[1]}, and {HN_cloth_prompts_list[2]}",
        f"{HN_naked_human_prompt} wearing {HN_cloth_prompts_list[0]}, {HN_cloth_prompts_list[1]}, {HN_cloth_prompts_list[2]}, and {HN_cloth_prompts_list[3]}",
        f"{HN_naked_human_prompt} wearing {HN_cloth_prompts_list[0]}, {HN_cloth_prompts_list[1]}, {HN_cloth_prompts_list[2]}, {HN_cloth_prompts_list[3]}, and {HN_cloth_prompts_list[4]}"
    ]
    RD_prompts_list = [cloth_prompt for cloth_prompt in cloth_prompts_list]
    neg_prompts_list=[
        "Tops, shirt, shoes, socks, hats, caps, gloves",
        "Tops, shirt, hats, caps, gloves",
        "",
        "",
        ""
    ]
    RD_neg_prompts_list=[
        "skin, head, hands, feet, faces, body, legs, arms, low quality",
        "skin, head, hands, feet, faces, body, legs, arms, low quality",
        "skin, head, hands, feet, faces, body, legs, arms, low quality",
        "skin, head, hands, feet, faces, body, legs, arms, low quality",
        "skin, head, hands, feet, faces, body, legs, arms, low quality"
    ]
    shape_init_list = [
        f"human:opt_smplx:dapose1:{name_list[1]}:1.007",
        f"human:opt_smplx:dapose1:{name_list[2]}:1.002_1.01",
        f"human:opt_smplx:dapose1:{name_list[3]}:1.01",
        f"human:opt_smplx:dapose1:{name_list[4]}:1.002_1.007",
        f"human:opt_smplx:dapose1:{name_list[5]}:1.002_1.007"
        ]
    asset_ckpt_list=[
        [],
        []
    ]
    asset_bbox_list = [
        [1.0,1.0,1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,0.0,0.0,0.0]
    ]
    asset_name_list = name_list
    clothed_human_tag_list=[prompt.replace(" ", "_").replace(".", "").replace(",", "") for prompt in HN_prompts_list]
    naked_human_tag = naked_human_prompt.replace(" ", "_").replace(".", "").replace(",", "")
    asset_ckpt_list[0].append(f"{naked_human_exp_root_dir}/{exp_name1}/{naked_human_tag}/ckpts/last.ckpt")
    asset_ckpt_list[1].append(f"{naked_human_exp_root_dir}/{exp_name3}/{naked_human_tag}/ckpts/last.ckpt")

    for cloth_idx in range(len(clothed_human_tag_list)):
        tag = clothed_human_tag_list[cloth_idx]
        prompt = HN_prompts_list[cloth_idx]
        RD_prompt = RD_prompts_list[cloth_idx]
        ori_prompt = HN_prompts_list[cloth_idx-1] if cloth_idx > 0 else naked_human_prompt
        neg_prompt = neg_prompts_list[cloth_idx]
        RD_neg_prompt = RD_neg_prompts_list[cloth_idx]
        data_part_ratio = data_part_ratio_list[cloth_idx]
        shape_init = shape_init_list[cloth_idx]
        asset_ckpt_list[0].append(f"{clothed_human_exp_root_dir}/{exp_name1}/{tag}/ckpts/last.ckpt")
        asset_ckpt_list[1].append(f"{clothed_human_exp_root_dir}/{exp_name2}/{tag}/ckpts/last.ckpt")
        cache_path1 = f"{clothed_human_exp_root_dir}/{exp_name1}/{tag}/save/cache"
        cache_path2 = f"{clothed_human_exp_root_dir}/{exp_name2}/{tag}/save/cache"

        command1 = ["python", "launch.py", "--config", "configs/cloth-geometry.yaml", "--gpu", f"{gpu_idx}",
                    "--train", f"tag={tag}", f"name={exp_name1}", f"exp_root_dir={clothed_human_exp_root_dir}",
                    f"data.fullbody_part_ratio.ratio={data_part_ratio}", 
                    f"system.test_save_path={cache_path1}", f"system.prompt_processor.prompt={prompt}, black background, normal map", 
                    f"system.prompt_processor_add.prompt={prompt}, black background, depth map", 
                    "system.prompt_processor.human_part_prompt=false", "system.geometry.use_sdf_loss=false", 
                    f"system.geometry.shape_init={shape_init}",
                    "system.geometry.shape_init_params=0.9", f"system.asset.ck_path={asset_ckpt_list[0][:cloth_idx+1]}", 
                    f"system.asset.bbox_info={asset_bbox_list[:cloth_idx+1]}", f"system.asset.name={asset_name_list[:cloth_idx+1]}"]
        command2 = ["python", "launch.py", "--config", "configs/cloth-texture-coarse.yaml", "--gpu", f"{gpu_idx}",
                    "--train", f"tag={tag}", f"name={exp_name2}", f"exp_root_dir={clothed_human_exp_root_dir}",
                    f"data.fullbody_part_ratio.ratio={data_part_ratio}", 
                    f"system.geometry_convert_from={asset_ckpt_list[0][cloth_idx + 1]}",
                    f"system.test_save_path={cache_path2}", f"system.prompt_processor.prompt={prompt}",
                    "system.prompt_processor.human_part_prompt=false", f"system.asset.ck_path={asset_ckpt_list[1][:cloth_idx+1]}", 
                    f"system.asset.bbox_info={asset_bbox_list[:cloth_idx+1]}", f"system.asset.name={asset_name_list[:cloth_idx+1]}"]
        command3 = ["python", "launch.py", "--config", "configs/cloth-texture-fine.yaml", "--gpu", f"{gpu_idx}",
                    "--train", f"tag={tag}", f"name={exp_name3}", f"exp_root_dir={clothed_human_exp_root_dir}",
                    f"data.dataroot={cache_path2}",  f"system.geometry_convert_from={asset_ckpt_list[1][cloth_idx + 1]}",
                    f"system.test_save_path={cache_path2}", f"system.prompt_processor.prompt={prompt}",
                    "system.prompt_processor.human_part_prompt=false", f"system.asset.ck_path={asset_ckpt_list[1][:cloth_idx+1]}", 
                    f"system.asset.bbox_info={asset_bbox_list[:cloth_idx+1]}", f"system.asset.name={asset_name_list[:cloth_idx+1]}"]

        if neg_prompt != "":
            command1.append(f"system.prompt_processor.negative_prompt={neg_prompt}")
            command1.append(f"system.prompt_processor_add.negative_prompt={neg_prompt}")
            command2.append(f"system.prompt_processor.negative_prompt={neg_prompt}")
            command3.append(f"system.prompt_processor.negative_prompt={neg_prompt}")

        if True:
            command1.append(f"system.prompt_processor_RD_StableDiffusion.prompt={RD_prompt}")
            command1.append(f"system.prompt_processor_RD_StableDiffusion.negative_prompt={RD_neg_prompt}")
            command2.append(f"system.prompt_processor_RD_StableDiffusion.prompt={RD_prompt}")
            command3.append(f"system.prompt_processor_RD_StableDiffusion.prompt={RD_prompt}")

        if cloth_idx == 0 or cloth_idx == 2:
            command1.append("system.geometry.isosurface_resolution=shell:0.05:0.9:2e-7")
            command2.append("system.geometry.isosurface_resolution=shell:0.05:0.9:2e-7")
            command3.append("system.geometry.isosurface_resolution=shell:0.05:0.9:2e-7")

            command1.extend(["system_type=humannorm-cloth-gshell-system", "system.geometry_type=implicit-gshell-cloth", "system.gshell_mode=nonwt", "system.geometry.msdf_type=implicit", "system.loss.lambda_temp=100", "system.loss.lambda_msdf_temp=100", "system.temp_loss_mode=bbox_mse", "system.msdf_temp_loss_mode=mse", "system.loss.lambda_collision=10000"])
            command2.extend(["system_type=humannorm-cloth-gshell-system", "system.geometry_type=tetrahedra-gshell-grid-cloth", "system.gshell_mode=nonwt"])
            command3.extend(["system_type=humannorm-cloth-gshell-system", "system.geometry_type=tetrahedra-gshell-grid-cloth", "system.gshell_mode=nonwt"])

        else:
            command1.append("system.geometry.isosurface_resolution=cube:256")
            command2.append("system.geometry.isosurface_resolution=cube:256")
            command3.append("system.geometry.isosurface_resolution=cube:256")

            command1.extend(["system.loss.lambda_temp=1000", "system.temp_loss_mode=in"])

        command1.append(f"trainer.max_steps=5000")
        command2.append(f"trainer.max_steps=2000")
        command3.append(f"trainer.max_steps=1000")

        tmp_str = ""
        for c in command1:
            tmp_str += c + " "
        print(tmp_str)
        if os.path.exists(f"{clothed_human_exp_root_dir}/{exp_name1}/{tag}/save/it5000-test-normal.mp4"):
            pass
        else:
            subprocess.run(["rm", "-rf", f"{clothed_human_exp_root_dir}/{exp_name1}/{tag}"])
            subprocess.run(command1)

        tmp_str = ""
        for c in command2:
            tmp_str += c + " "
        print(tmp_str)
        if os.path.exists(f"{clothed_human_exp_root_dir}/{exp_name2}/{tag}/save/it2000-test-normal.mp4"):
            pass
        else:
            subprocess.run(["rm", "-rf", f"{clothed_human_exp_root_dir}/{exp_name2}/{tag}"])
            subprocess.run(command2)

        if cloth_idx == (len(clothed_human_tag_list) - 1):
            command3.append(f"system.exporter.save_flag=true")
            tmp_str = ""
            for c in command3:
                tmp_str += c + " "
            print(tmp_str)
            if os.path.exists(f"{clothed_human_exp_root_dir}/{exp_name3}/{tag}/save/it1000-test-normal.mp4"):
                pass
            else:
                subprocess.run(["rm", "-rf", f"{clothed_human_exp_root_dir}/{exp_name3}/{tag}"])
                subprocess.run(command3)