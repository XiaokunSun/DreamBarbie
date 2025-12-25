from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import math, random, copy
import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
import numpy as np
from PIL import Image

@threestudio.register("nvdiff-rasterizer-multiasset")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        normal_type: str = 'world'
        use_sdf_loss: bool = False
        rgb_type: str = "color"  # albedo or color

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
        asset_geometry_list, 
        asset_material_list, 
        asset_background_list,
        asset_system_cfg_list,
        asset_cfg,
        cloth_cfg
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        self.asset_geometry_list = asset_geometry_list
        self.asset_material_list = asset_material_list
        self.asset_background_list = asset_background_list
        self.asset_system_cfg_list = asset_system_cfg_list
        self.asset_num = len(asset_geometry_list)
        self.asset_cfg = asset_cfg
        self.cloth_cfg = cloth_cfg

    def render_geometry(self, verts, faces, normals, c2w, height, width):
        rast, _ = self.ctx.rasterize(verts, faces, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, verts, faces)
        gb_normal, _ = self.ctx.interpolate_one(normals, rast, faces)

        if self.cfg.normal_type == 'world':
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif self.cfg.normal_type == 'camera':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            bg_normal = torch.zeros_like(gb_normal)
            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_ =  torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif self.cfg.normal_type == 'controlnet':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            # nerf coord to a special coord for controlnet
            gb_normal = torch.cat([-gb_normal[:,:,:,0:1], gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3]], -1)
            bg_normal = torch.zeros_like(gb_normal)
            bg_normal[~mask[..., 0]] = torch.Tensor([[126/255, 107/255, 1.0]]).to(bg_normal)

            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        else:
            raise ValueError(f"Unknown normal type: {self.cfg.normal_type}")

        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, verts, faces
        )

        # gb_depth = rast[..., 2:3]
        # gb_depth = 1./(gb_depth + 1e-7)
        gb_depth, _ = self.ctx.interpolate_one(verts[0,:, :3].contiguous(), rast, faces)
        gb_depth_ = gb_depth[..., 2:3].clone()
        gb_depth_[~mask[..., 0]] = 1e7
        gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)


        if mask.sum() == 0:
            gb_depth_aa = torch.ones_like(gb_depth)
        else:
            max_depth = torch.max(gb_depth[mask[..., 0]])
            min_depth = torch.min(gb_depth[mask[..., 0]])
            gb_depth_aa = torch.lerp(
                    torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
                )
            gb_depth_aa = self.ctx.antialias(
                gb_depth_aa, rast, verts, faces
            )
        return mask_aa, gb_normal_aa, gb_depth_aa, rast, mask, gb_normal_, gb_normal, gb_depth_

    def render_texture(self, mesh, rast, raw_mask, raw_normal, camera_positions, light_positions, height, width, geometry, material, background, v_pos_clip, scale=None, trans=None):
        selector = raw_mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
        gb_viewdirs = F.normalize(
            gb_pos - camera_positions[:, None, None, :], dim=-1
        )
        gb_light_positions = light_positions[:, None, None, :].expand(
            -1, height, width, -1
        )
        if scale is not None and trans is not None:
            positions = gb_pos[selector]
            positions = (positions - trans) / scale
        else:
            positions = gb_pos[selector]
        geo_out = geometry(positions, output_normal=False)
        extra_geo_info = {}
        if material.requires_normal:
            extra_geo_info["shading_normal"] = raw_normal[selector]
        if material.requires_tangent:
            tmp = mesh.v_tng
            gb_tangent, _ = self.ctx.interpolate_one(
                mesh.v_tng, rast, mesh.t_pos_idx
            )
            gb_tangent = F.normalize(gb_tangent, dim=-1)
            extra_geo_info["tangent"] = gb_tangent[selector]
        rgb_fg = material(
            viewdirs=gb_viewdirs[selector],
            positions=positions,
            light_positions=gb_light_positions[selector],
            **extra_geo_info,
            **geo_out
        )
        if type(rgb_fg) == dict:
            rgb_fg = rgb_fg[self.cfg.rgb_type]
        gb_rgb_fg = torch.zeros(v_pos_clip.shape[0], height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg
        gb_rgb_bg = background(dirs=gb_viewdirs)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, raw_mask.float())
        gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
        return gb_rgb_fg, gb_rgb_bg, gb_rgb, gb_rgb_aa


    def set_asset(self, render_asset_idx, mvp_mtx):
        asset_verts_list = []
        asset_faces_list = []
        asset_normals_list = []
        asset_mesh_list = []
        asset_verts_num = 0
        for i in render_asset_idx:
            if hasattr(self.asset_geometry_list[i], 'msdf') or hasattr(self.asset_geometry_list[i], 'msdf_network'):
                asset_mesh, extra = self.asset_geometry_list[i].isosurface("nonwt")
            else:
                asset_mesh = self.asset_geometry_list[i].isosurface()
            # if self.render_rgb:
            #     tmp = asset_mesh.v_tng
            if self.asset_cfg.name[i] == "human":
                scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device())
                trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device())
            else:
                scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device()) * self.asset_geometry_list[i].scale
                trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device()) + self.asset_geometry_list[i].trans
            asset_mesh_precessed = copy.copy(asset_mesh)
            asset_mesh_precessed.v_pos = (asset_mesh_precessed.v_pos * scale) + trans
            asset_verts_list.append(self.ctx.vertex_transform(asset_mesh_precessed.v_pos, mvp_mtx))
            asset_faces_list.append(asset_mesh_precessed.t_pos_idx + asset_verts_num)
            asset_normals_list.append(asset_mesh_precessed.v_nrm)
            asset_mesh_list.append(asset_mesh_precessed)
            asset_verts_num += asset_mesh_precessed.v_pos.shape[0]
        self.asset_verts = torch.cat(asset_verts_list, 1)
        self.asset_faces = torch.cat(asset_faces_list, 0)
        self.asset_normals = torch.cat(asset_normals_list, 0)
        self.asset_mesh_list = asset_mesh_list
        self.asset_verts_num = asset_verts_num

    def set_cloth_c(self, mvp_mtx):
        self.extra = None
        if hasattr(self.geometry, 'msdf') or hasattr(self.geometry, 'msdf_network'):
            if self.gshell_mode == "wt":
                cloth_c_mesh = self.geometry.isosurface(self.gshell_mode)
            else:
                cloth_c_mesh, self.extra = self.geometry.isosurface(self.gshell_mode)
        else:
            cloth_c_mesh = self.geometry.isosurface()
        cloth_c_sdf_loss = None
        # if self.render_rgb:
        #     tmp = cloth_c_mesh.v_tng
        self.cloth_c_verts = self.ctx.vertex_transform(cloth_c_mesh.v_pos, mvp_mtx)
        self.cloth_c_faces = cloth_c_mesh.t_pos_idx
        self.cloth_c_normals = cloth_c_mesh.v_nrm
        self.cloth_c_mesh = cloth_c_mesh
        self.cloth_c_sdf_loss = cloth_c_sdf_loss

    def set_cloth_a(self, mvp_mtx, reset_cloth_c=True):
        if reset_cloth_c: 
            self.set_cloth_c(mvp_mtx) 
        scale = self.geometry.scale
        trans = self.geometry.trans
        cloth_a_mesh = copy.copy(self.cloth_c_mesh)
        cloth_a_mesh.v_pos = (cloth_a_mesh.v_pos * scale) + trans
        self.cloth_a_verts = self.ctx.vertex_transform(cloth_a_mesh.v_pos, mvp_mtx)
        self.cloth_a_faces = cloth_a_mesh.t_pos_idx
        self.cloth_a_normals = cloth_a_mesh.v_nrm
        self.cloth_a_mesh = cloth_a_mesh

    def set_comp(self, render_asset_idx, mvp_mtx, reset_cloth_a=True, reset_cloth_c=True, reset_asset=True):
        if reset_asset:
            self.set_asset(render_asset_idx, mvp_mtx)
        if reset_cloth_a:
            self.set_cloth_a(mvp_mtx, reset_cloth_c)
        self.comp_verts = torch.cat([self.asset_verts, self.cloth_a_verts], 1)
        self.comp_faces = torch.cat([self.asset_faces, self.cloth_a_faces + self.asset_verts_num], 0)
        self.comp_normals = torch.cat([self.asset_normals, self.cloth_a_normals], 0)
        self.comp_mesh_list = self.asset_mesh_list + [self.cloth_a_mesh]

    def render_mesh(
        self,
        cfg,
        geometry,
        material,
        background,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        gshell_mode: str = "nonwt"
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        if hasattr(geometry, 'msdf') or hasattr(geometry, 'msdf_network'):
            if gshell_mode == "wt":
                mesh = geometry.isosurface(gshell_mode)
            else:
                mesh, extra = geometry.isosurface(gshell_mode)
        else:
            mesh = geometry.isosurface()
        sdf_loss = None

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh, 'sdf_loss': sdf_loss}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)

        if cfg.normal_type == 'world':
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif cfg.normal_type == 'camera':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            bg_normal = torch.zeros_like(gb_normal)
            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        elif cfg.normal_type == 'controlnet':
            # world coord to cam coord
            gb_normal = gb_normal.view(-1, height*width, 3)
            gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
            gb_normal = gb_normal.view(-1, height, width, 3)
            gb_normal = F.normalize(gb_normal, dim=-1)
            # nerf coord to a special coord for controlnet
            gb_normal = torch.cat([-gb_normal[:,:,:,0:1], gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3]], -1)
            bg_normal = torch.zeros_like(gb_normal)
            bg_normal[~mask[..., 0]] = torch.Tensor([[126/255, 107/255, 1.0]]).to(bg_normal)

            gb_normal_aa = torch.lerp(
                bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
            )
        else:
            raise ValueError(f"Unknown normal type: {cfg.normal_type}")

        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"normal": gb_normal_aa})  # in [0, 1]

        # gb_depth = rast[..., 2:3]
        # gb_depth = 1./(gb_depth + 1e-7)
        gb_depth, _ = self.ctx.interpolate_one(v_pos_clip[0,:, :3].contiguous(), rast, mesh.t_pos_idx)
        gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
        max_depth = torch.max(gb_depth[mask[..., 0]])
        min_depth = torch.min(gb_depth[mask[..., 0]])
        gb_depth_aa = torch.lerp(
                torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
            )
        gb_depth_aa = self.ctx.antialias(
            gb_depth_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"depth":gb_depth_aa})  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = geometry(positions, output_normal=False)

            extra_geo_info = {}
            if material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if material.requires_tangent:
                tmp = mesh.v_tng
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            rgb_fg = material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"rgb": gb_rgb_aa, "rgb_bg": gb_rgb_bg})
        return out
        
    def render_human_and_cloth(        
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        gshell_mode="nonwt",
        **kwargs):
        out = []
        for i in range(self.asset_num):
            render_result = self.render_mesh(cfg=self.asset_system_cfg_list[i].renderer, geometry=self.asset_geometry_list[i], material=self.asset_material_list[i], background=self.asset_background_list[i], mvp_mtx=mvp_mtx, c2w=c2w, camera_positions=camera_positions, light_positions=light_positions, height=height, width=width, render_rgb=render_rgb, gshell_mode=gshell_mode)
            out.append(render_result)
        render_result = self.render_mesh(cfg=self.cfg, geometry=self.geometry, material=self.material, background=self.background, mvp_mtx=mvp_mtx, c2w=c2w, camera_positions=camera_positions, light_positions=light_positions, height=height, width=width, render_rgb=render_rgb, gshell_mode=gshell_mode)
        out.append(render_result)
        return out

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        render_mode="cloth_c+comp",
        render_asset_idx=None,
        gshell_mode="nonwt",
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        if render_asset_idx is None:
            render_asset_idx = list(range(self.asset_num))  
        out = {}
        self.render_rgb = render_rgb
        self.gshell_mode = gshell_mode
        if not render_rgb:
            self.cloth_c_sdf_loss = None
            if "asset" in render_mode:
                self.set_asset(render_asset_idx=render_asset_idx, mvp_mtx=mvp_mtx)
                asset_render_mask, asset_render_normal, asset_render_depth, _, _, _, _, _ = self.render_geometry(self.asset_verts, self.asset_faces, self.asset_normals, c2w, height, width)
                out.update({"asset_mesh_list": self.asset_mesh_list, "asset_render_mask":asset_render_mask, "asset_render_normal":asset_render_normal, "asset_render_depth":asset_render_depth})

            if "cloth_c" in render_mode:
                self.set_cloth_c(mvp_mtx)
                cloth_c_render_mask, cloth_c_render_normal, cloth_c_render_depth, _, _, _, _, _ = self.render_geometry(self.cloth_c_verts, self.cloth_c_faces, self.cloth_c_normals, c2w, height, width)
                out.update({"cloth_c_mesh": self.cloth_c_mesh, "cloth_c_render_mask":cloth_c_render_mask, "cloth_c_render_normal":cloth_c_render_normal, "cloth_c_render_depth":cloth_c_render_depth})

            if "cloth_a" in render_mode:
                self.set_cloth_a(mvp_mtx, reset_cloth_c=("cloth_c" not in render_mode))
                cloth_a_render_mask, cloth_a_render_normal, cloth_a_render_depth, _, _, _, _, _ = self.render_geometry(self.cloth_a_verts, self.cloth_a_faces, self.cloth_a_normals, c2w, height, width)
                out.update({"cloth_a_mesh": self.cloth_a_mesh, "cloth_a_render_mask":cloth_a_render_mask, "cloth_a_render_normal":cloth_a_render_normal, "cloth_a_render_depth":cloth_a_render_depth})

            if "comp" in render_mode:
                self.set_comp(render_asset_idx=render_asset_idx, mvp_mtx=mvp_mtx, reset_cloth_a=("cloth_a" not in render_mode), reset_cloth_c=("cloth_c" not in render_mode), reset_asset=("asset" not in render_mode))
                comp_render_mask, comp_render_normal, comp_render_depth, _, _, _, _, _ = self.render_geometry(self.comp_verts, self.comp_faces, self.comp_normals, c2w, height, width)
                out.update({"comp_mesh_list": self.comp_mesh_list, "comp_render_mask":comp_render_mask, "comp_render_normal":comp_render_normal, "comp_render_depth":comp_render_depth})
            out.update({"cloth_c_sdf_loss": self.cloth_c_sdf_loss})
        else:
            raw_mask_list = []
            raw_normal_list = []
            raw_depth_list = []
            raw_rgb_list = []
            
            if "asset" in render_mode:
                for i in render_asset_idx:
                    self.set_asset(render_asset_idx=[i], mvp_mtx=mvp_mtx)
                    tmp_rast, _ = self.ctx.rasterize(self.asset_verts, self.asset_faces, (height, width))
                    if (tmp_rast[..., 3:] > 0).sum() == 0:
                        continue
                    asset_render_mask, asset_render_normal, asset_render_depth, asset_rast, asset_raw_mask, asset_raw_normal, asset_gb_normal, asset_raw_depth = self.render_geometry(self.asset_verts, self.asset_faces, self.asset_normals, c2w, height, width)
                    if self.asset_cfg.name[i] == "human":
                        scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device())
                        trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device())
                    else:
                        scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device()) * self.asset_geometry_list[i].scale
                        trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device()) + self.asset_geometry_list[i].trans
                    _, _, gb_rgb, _ = self.render_texture(self.asset_mesh_list[0], asset_rast, asset_raw_mask, asset_gb_normal, camera_positions, light_positions, height, width, self.asset_geometry_list[i], self.asset_material_list[i], self.asset_background_list[0], self.asset_verts, scale, trans)
                    raw_mask_list.append(asset_raw_mask)
                    raw_normal_list.append(asset_raw_normal)
                    raw_depth_list.append(asset_raw_depth)
                    raw_rgb_list.append(gb_rgb)
                total_raw_mask = torch.cat(raw_mask_list, dim=-1)
                total_raw_normal = torch.cat(raw_normal_list, dim=-1)
                total_raw_depth = torch.cat(raw_depth_list, dim=-1)
                total_raw_rgb = torch.cat(raw_rgb_list, dim=-1)
                z_min_index = torch.argmin(total_raw_depth, dim=-1, keepdim=True)
                total_raw_rgb = total_raw_rgb.reshape(total_raw_rgb.shape[:-1] + (-1, 3))
                asset_render_rgb = torch.gather(total_raw_rgb, -2, z_min_index[..., None].repeat(1,1,1,1,3)).squeeze(-2)
                total_raw_normal = total_raw_normal.reshape(total_raw_normal.shape[:-1] + (-1, 3))
                asset_render_normal = torch.gather(total_raw_normal, -2, z_min_index[..., None].repeat(1,1,1,1,3)).squeeze(-2)
                self.set_asset(render_asset_idx=render_asset_idx, mvp_mtx=mvp_mtx)
                asset_render_mask, _, asset_render_depth, asset_rast, _, _, _, _ = self.render_geometry(self.asset_verts, self.asset_faces, self.asset_normals, c2w, height, width)
                asset_render_rgb = self.ctx.antialias(asset_render_rgb, asset_rast, self.asset_verts, self.asset_faces)    
                asset_render_normal = self.ctx.antialias(asset_render_normal, asset_rast, self.asset_verts, self.asset_faces)    
                out.update({"asset_mesh_list": self.asset_mesh_list, "asset_render_mask":asset_render_mask, "asset_render_normal":asset_render_normal, "asset_render_depth":asset_render_depth, "asset_render_rgb":asset_render_rgb})
            if "cloth_c" in render_mode:
                self.set_cloth_c(mvp_mtx)
                cloth_c_render_mask, cloth_c_render_normal, cloth_c_render_depth, cloth_c_rast, cloth_c_raw_mask, cloth_c_raw_normal, cloth_c_gb_normal, cloth_c_raw_depth = self.render_geometry(self.cloth_c_verts, self.cloth_c_faces, self.cloth_c_normals, c2w, height, width)
                _, _, _, cloth_c_render_rgb = self.render_texture(self.cloth_c_mesh, cloth_c_rast, cloth_c_raw_mask, cloth_c_gb_normal, camera_positions, light_positions, height, width, self.geometry, self.material, self.background, self.cloth_c_verts)
                out.update({"cloth_c_mesh": self.cloth_c_mesh, "cloth_c_render_mask":cloth_c_render_mask, "cloth_c_render_normal":cloth_c_render_normal, "cloth_c_render_depth":cloth_c_render_depth, "cloth_c_render_rgb":cloth_c_render_rgb})
            if "cloth_a" in render_mode:
                self.set_cloth_a(mvp_mtx, reset_cloth_c=("cloth_c" not in render_mode))
                cloth_a_render_mask, cloth_a_render_normal, cloth_a_render_depth, cloth_a_rast, cloth_a_raw_mask, cloth_a_raw_normal, cloth_a_gb_normal, cloth_a_raw_depth = self.render_geometry(self.cloth_a_verts, self.cloth_a_faces, self.cloth_a_normals, c2w, height, width)
                _, _, _, cloth_a_render_rgb = self.render_texture(self.cloth_a_mesh, cloth_a_rast, cloth_a_raw_mask, cloth_a_gb_normal, camera_positions, light_positions, height, width, self.geometry, self.material, self.background, self.cloth_a_verts, self.geometry.scale, self.geometry.trans)
                out.update({"cloth_a_mesh": self.cloth_a_mesh, "cloth_a_render_mask":cloth_a_render_mask, "cloth_a_render_normal":cloth_a_render_normal, "cloth_a_render_depth":cloth_a_render_depth, "cloth_a_render_rgb":cloth_a_render_rgb})
            if "comp" in render_mode:
                if "asset" not in render_mode:
                    for i in render_asset_idx:
                        self.set_asset(render_asset_idx=[i], mvp_mtx=mvp_mtx)
                        tmp_rast, _ = self.ctx.rasterize(self.asset_verts, self.asset_faces, (height, width))
                        if (tmp_rast[..., 3:] > 0).sum() == 0:
                            continue
                        asset_render_mask, asset_render_normal, asset_render_depth, asset_rast, asset_raw_mask, asset_raw_normal, asset_gb_normal, asset_raw_depth = self.render_geometry(self.asset_verts, self.asset_faces, self.asset_normals, c2w, height, width)
                        if self.asset_cfg.name[i] == "human":
                            scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device())
                            trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device())
                        else:
                            scale = torch.tensor(self.asset_cfg.bbox_info[i][:3]).float().to(get_device()) * self.asset_geometry_list[i].scale
                            trans = torch.tensor(self.asset_cfg.bbox_info[i][3:]).float().to(get_device()) + self.asset_geometry_list[i].trans
                        _, _, gb_rgb, _ = self.render_texture(self.asset_mesh_list[0], asset_rast, asset_raw_mask, asset_gb_normal, camera_positions, light_positions, height, width, self.asset_geometry_list[i], self.asset_material_list[i], self.asset_background_list[0], self.asset_verts, scale, trans)
                        raw_mask_list.append(asset_raw_mask)
                        raw_normal_list.append(asset_raw_normal)
                        raw_depth_list.append(asset_raw_depth)
                        raw_rgb_list.append(gb_rgb)
                self.set_cloth_a(mvp_mtx, reset_cloth_c=("cloth_c" not in render_mode))
                cloth_a_render_mask, cloth_a_render_normal, cloth_a_render_depth, cloth_a_rast, cloth_a_raw_mask, cloth_a_raw_normal, cloth_a_gb_normal, cloth_a_raw_depth = self.render_geometry(self.cloth_a_verts, self.cloth_a_faces, self.cloth_a_normals, c2w, height, width)
                _, _, gb_rgb, _ = self.render_texture(self.cloth_a_mesh, cloth_a_rast, cloth_a_raw_mask, cloth_a_gb_normal, camera_positions, light_positions, height, width, self.geometry, self.material, self.background, self.cloth_a_verts, self.geometry.scale, self.geometry.trans)
                raw_mask_list.append(cloth_a_raw_mask)
                raw_normal_list.append(cloth_a_raw_normal)
                raw_depth_list.append(cloth_a_raw_depth)
                raw_rgb_list.append(gb_rgb)
                total_raw_mask = torch.cat(raw_mask_list, dim=-1)
                total_raw_normal = torch.cat(raw_normal_list, dim=-1)
                total_raw_depth = torch.cat(raw_depth_list, dim=-1)
                total_raw_rgb = torch.cat(raw_rgb_list, dim=-1)
                z_min_index = torch.argmin(total_raw_depth, dim=-1, keepdim=True)
                total_raw_rgb = total_raw_rgb.reshape(total_raw_rgb.shape[:-1] + (-1, 3))
                comp_render_rgb = torch.gather(total_raw_rgb, -2, z_min_index[..., None].repeat(1,1,1,1,3)).squeeze(-2)
                total_raw_normal = total_raw_normal.reshape(total_raw_normal.shape[:-1] + (-1, 3))
                comp_render_normal = torch.gather(total_raw_normal, -2, z_min_index[..., None].repeat(1,1,1,1,3)).squeeze(-2)
                self.set_comp(render_asset_idx=render_asset_idx, mvp_mtx=mvp_mtx, reset_cloth_a=("cloth_a" not in render_mode), reset_cloth_c=("cloth_c" not in render_mode), reset_asset=("asset" not in render_mode))
                comp_render_mask, _, comp_render_depth, comp_rast, _, _, _, _ = self.render_geometry(self.comp_verts, self.comp_faces, self.comp_normals, c2w, height, width)
                comp_render_rgb = self.ctx.antialias(comp_render_rgb, comp_rast, self.comp_verts, self.comp_faces)    
                comp_render_normal = self.ctx.antialias(comp_render_normal, comp_rast, self.comp_verts, self.comp_faces)    
                out.update({"comp_mesh_list": self.comp_mesh_list, "comp_render_mask":comp_render_mask, "comp_render_normal":comp_render_normal, "comp_render_depth":comp_render_depth, "comp_render_rgb":comp_render_rgb})
        out.update({"extra": self.extra})
        return out
