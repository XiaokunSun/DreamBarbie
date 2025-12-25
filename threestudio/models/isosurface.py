import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *
import tetgen
import pyvista as pv
import pymeshlab
import trimesh
import os

class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError


class MarchingCubeCPUHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        import mcubes

        self.mc_func: Callable = mcubes.marching_cubes
        self._grid_vertices: Optional[Float[Tensor, "N3 3"]] = None
        self._dummy: Float[Tensor, "..."]
        self.register_buffer(
            "_dummy", torch.zeros(0, dtype=torch.float32), persistent=False
        )

    @property
    def grid_vertices(self) -> Float[Tensor, "N3 3"]:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            threestudio.warn(
                f"{self.__class__.__name__} does not support deformation. Ignoring."
            )
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(
            level.detach().cpu().numpy(), 0.0
        )  # transform to numpy
        v_pos, t_pos_idx = (
            torch.from_numpy(v_pos).float().to(self._dummy.device),
            torch.from_numpy(t_pos_idx.astype(np.int64)).long().to(self._dummy.device),
        )  # transform back to torch tensor on CUDA
        v_pos = v_pos / (self.resolution - 1.0)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, tets_path: str):
        super().__init__()
        self.resolution = resolution
        self.tets_path = tets_path

        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )

        tets = np.load(self.tets_path)
        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(tets["vertices"]).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(tets["indices"]).long(), persistent=False
        )

        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )

        return mesh

class MarchingTetrahedraShellHelper(IsosurfaceHelper):
    def __init__(self, mesh: Mesh, tet_shell_offset: float, tet_shell_decimate: float, tet_grid_volume: float, save_dir_path: str):
        super().__init__()
        if os.path.exists(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz")):
            tets = np.load(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz"))
            print("load tets from file: {}".format(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz")))
            vertices = tets["vertices"]
            indices = tets["indices"]
        else:
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
            ms.generate_resampled_uniform_mesh(offset=pymeshlab.AbsoluteValue(tet_shell_offset))
            ms.save_current_mesh(os.path.join(save_dir_path, "tmp_shell.obj"))
            mesh = pv.read(os.path.join(save_dir_path, "tmp_shell.obj"))
            downsampled_mesh = mesh.decimate(tet_shell_decimate)
            tet = tetgen.TetGen(downsampled_mesh)
            mesh_count = len(trimesh.load(os.path.join(save_dir_path, "tmp_shell.obj")).split(only_watertight=False))
            if mesh_count == 1:
                tet.make_manifold(verbose=True)
            vertices, indices = tet.tetrahedralize(fixedvolume=1, 
                                            maxvolume=tet_grid_volume, 
                                            regionattrib=1, 
                                            nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
            shell = tet.grid.extract_surface()
            shell.save(os.path.join(save_dir_path, "dmtet_shell.ply"))
            vertices = vertices / 2 + 0.5
            np.savez(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz"), vertices=vertices, indices=indices)

        print('tet_shell_offset: {}, tet_shell_decimate: {}, tet_grid_volume: {}, shape of vertices: {}, shape of grids: {}'.format(tet_shell_offset, tet_shell_decimate, tet_grid_volume, vertices.shape, indices.shape))

        self.resolution = None

        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )

        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(vertices).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(indices).long(), persistent=False
        )

        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support deformation."
            )
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )

        return mesh
    
class GshellHelper(IsosurfaceHelper):
    def __init__(self, resolution=None, tets_path=None, mesh=None, tet_shell_offset=None, tet_shell_decimate=None, tet_grid_volume=None, save_dir_path=None):
        super().__init__()

        if resolution is None and tets_path is None:
            if os.path.exists(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz")):
                tets = np.load(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz"))
                print("load tets from file: {}".format(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz")))
                vertices = tets["vertices"]
                indices = tets["indices"]
            else:
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
                ms.generate_resampled_uniform_mesh(offset=pymeshlab.AbsoluteValue(tet_shell_offset))
                ms.save_current_mesh(os.path.join(save_dir_path, "tmp_shell.obj"))
                mesh = pv.read(os.path.join(save_dir_path, "tmp_shell.obj"))
                downsampled_mesh = mesh.decimate(tet_shell_decimate)
                tet = tetgen.TetGen(downsampled_mesh)
                mesh_count = len(trimesh.load(os.path.join(save_dir_path, "tmp_shell.obj")).split(only_watertight=False))
                if mesh_count == 1:
                    tet.make_manifold(verbose=True)
                vertices, indices = tet.tetrahedralize(fixedvolume=1, 
                                                maxvolume=tet_grid_volume, 
                                                regionattrib=1, 
                                                nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
                shell = tet.grid.extract_surface()
                shell.save(os.path.join(save_dir_path, "dmtet_shell.ply"))
                vertices = vertices / 2 + 0.5
                np.savez(os.path.join(save_dir_path, f"tets_{tet_shell_offset}_{tet_shell_decimate}_{tet_grid_volume}.npz"), vertices=vertices, indices=indices)

            print('tet_shell_offset: {}, tet_shell_decimate: {}, tet_grid_volume: {}, shape of vertices: {}, shape of grids: {}'.format(tet_shell_offset, tet_shell_decimate, tet_grid_volume, vertices.shape, indices.shape))
            self.resolution = None
        else:
            self.resolution = resolution
            self.tets_path = tets_path
            tets = np.load(self.tets_path)
            vertices, indices = tets["vertices"], tets["indices"]

        self.triangle_table: Integer[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )
        self.mesh_edge_table: Integer[Tensor, "..."]
        self.register_buffer(
            "mesh_edge_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [ 1,  0,  2,  1, -1, -1],
                    [ 4,  0,  3,  4, -1, -1],
                    [ 1,  3,  4,  2,  1, -1],
                    [ 3,  1,  5,  3, -1, -1],
                    [ 2,  5,  3,  0,  2, -1],
                    [ 1,  5,  4,  0,  1, -1],
                    [ 4,  2,  5,  4, -1, -1],
                    [ 4,  5,  2,  4, -1, -1],
                    [ 4,  5,  1,  0,  4, -1],
                    [ 3,  5,  2,  0,  3, -1],
                    [ 1,  3,  5,  1, -1, -1],
                    [ 4,  3,  1,  2,  4, -1],
                    [ 3,  0,  4,  3, -1, -1],
                    [ 2,  0,  1,  2, -1, -1],
                    [-1, -1, -1, -1, -1, -1]
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.triangle_table_tri: Integer[Tensor, "..."]
        self.register_buffer(
            "triangle_table_tri",
            torch.as_tensor(
                [
                ## 000
                    [-1, -1, -1, -1, -1, -1],
                ## 001
                    [ 4,  2,  5, -1, -1, -1],
                ## 010
                    [ 3,  1,  4, -1, -1, -1],
                ## 011
                    [ 3,  1,  2,  3,  2,  5],
                ## 100
                    [ 0,  3,  5, -1, -1, -1],
                ## 101
                    [ 0,  3,  4,  0,  4,  2],
                ## 110
                    [ 0,  1,  4,  0,  4,  5],
                ## 111
                    [ 0,  1,  2, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.triangle_table_quad: Integer[Tensor, "..."]
        self.register_buffer(
            "triangle_table_quad",
            torch.as_tensor(
                [
                ### in the order of [0, 1, 2, 3]
                ### so 1000 corresponds to single positive mSDF vertex of index 0
                ## 0000
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                ## 0001
                    [ 6,  3,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                ## 0010
                    [ 5,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                ## 0011
                    [ 5,  2,  7,  3,  7,  2, -1, -1, -1, -1, -1, -1],
                ## 0100
                    [ 4,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                ## 0101
                    [ 4,  1,  5,  4,  5,  7,  5,  6,  7,  7,  6,  3],
                ## 0110
                    [ 4,  1,  2,  6,  4,  2, -1, -1, -1, -1, -1, -1],
                ## 0111
                    [ 4,  1,  2,  7,  4,  2,  7,  2,  3, -1, -1, -1],
                ## 1000
                    [ 0,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                ## 1001
                    [ 0,  4,  6,  3,  0,  6, -1, -1, -1, -1, -1, -1],
                ## 1010
                    [ 0,  4,  5,  0,  5,  2,  0,  2,  6,  0,  6,  7],
                ## 1011
                    [ 0,  4,  5,  0,  5,  2,  0,  2,  3, -1, -1, -1],
                ## 1100
                    [ 0,  1,  5,  7,  0,  5, -1, -1, -1, -1, -1, -1],
                ## 1101
                    [ 0,  1,  5,  0,  5,  6,  0,  6,  3, -1, -1, -1],
                ## 1110
                    [ 0,  1,  2,  0,  2,  6,  0,  6,  7, -1, -1, -1],
                ## 1111
                    [ 0,  1,  2,  0,  2,  3, -1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_tri_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_tri_table",
            torch.as_tensor(
                [0,1,1,2,1,2,2,1],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_quad_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_quad_table",
            torch.as_tensor(
                [0,1,1,2,1,4,2,3,1,2,4,3,2,3,3,2],
                dtype=torch.long,
            ),
            persistent=False,
        )

        edge_ind_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        msdf_from_tetverts = []
        for i in range(5):
            for j in range(i+1, 6):
                if (edge_ind_list[i][0] == edge_ind_list[j][0]
                    or edge_ind_list[i][0] == edge_ind_list[j][1]
                    or edge_ind_list[i][1] == edge_ind_list[j][0]
                    or edge_ind_list[i][1] == edge_ind_list[j][1]
                ):
                    msdf_from_tetverts.extend([edge_ind_list[i][0], edge_ind_list[i][1], edge_ind_list[j][0], edge_ind_list[j][1]])

        self.msdf_from_tetverts = torch.tensor(msdf_from_tetverts)

        
        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(vertices).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(indices).long(), persistent=False
        )

        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def _forward_gshell(self, pos_nx3, sdf_n, msdf_n, tet_fx4, output_watertight_template=True):
        sdf_n = sdf_n.float()
        with torch.no_grad():
            ### To determine if tets are valid
            ### Step 1: SDF criteria
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)


            ### Step 2: pre-filtering with mSDF - mSDF cannot be all non-negative
            msdf_fx4 = msdf_n[tet_fx4.reshape(-1)].reshape(-1,4)
            msdf_sign_fx4 = msdf_fx4 > 0
            msdf_sign_sum = torch.sum(msdf_sign_fx4, -1)

            if output_watertight_template:
                valid_tets = (occ_sum>0) & (occ_sum<4) 
            else:
                valid_tets = (occ_sum>0) & (occ_sum<4) & (msdf_sign_sum > 0)

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)  

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=pos_nx3.device)
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim = True)
        denominator = torch.sign(denominator) * (denominator.abs() + 1e-12)
        denominator[denominator == 0] = 1e-12

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        msdf_to_interp = msdf_n[interp_v.reshape(-1)].reshape(-1,2)
        msdf_vert = (msdf_to_interp * edges_to_interp_sdf.squeeze(-1)).sum(1)
        msdf_vert_stopvgd = (msdf_to_interp * edges_to_interp_sdf.squeeze(-1).detach()).sum(1)

        # (M, 6), M: num of pre-filtered tets, storing indices (besides -1) from 0 to num_mask_edges
        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        # triangle count
        num_triangles = self.num_triangles_table[tetindex]

        # Get global face index (static, does not depend on topology), before mSDF processing
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device=pos_nx3.device)[valid_tets]
        face_gidx_pre = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        # Generate triangle indices before msdf processing
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)
        
        ###### Triangulation with mSDF
        ### Note: we allow area-0 triangular faces for convenience. Can always remove them during post-processing
        with torch.no_grad():
            mesh_edge_tri = torch.gather(input=idx_map[num_triangles == 1], dim=1, 
                    index=self.mesh_edge_table[tetindex[num_triangles == 1]][:, [0, 1, 1, 2, 2, 0]]
                ).view(-1, 3, 2)
            mesh_edge_quad = torch.gather(input=idx_map[num_triangles == 2], dim=1, 
                    index=self.mesh_edge_table[tetindex[num_triangles == 2]][:, [0, 1, 1, 2, 2, 3, 3, 0]]
                ).view(-1, 4, 2)
            mocc_fx3 = (msdf_vert[mesh_edge_tri[:, :, 0].reshape(-1)].reshape(-1, 3) > 0).long()
            mocc_fx4 = (msdf_vert[mesh_edge_quad[:, :, 0].reshape(-1)].reshape(-1, 4) > 0).long()

        ### Attributes to be interpolated for (non-watertight) mesh vertices on the boundary
        edges_to_interp_vpos_tri = verts[mesh_edge_tri.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_vpos_quad = verts[mesh_edge_quad.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_msdf_tri = msdf_vert[mesh_edge_tri.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_quad = msdf_vert[mesh_edge_quad.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_tri_stopvgd = msdf_vert_stopvgd[mesh_edge_tri.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_msdf_quad_stopvgd = msdf_vert_stopvgd[mesh_edge_quad.reshape(-1)].reshape(-1,2,1)

        ### Linear interpolation on mesh edges (triangle / quad faces)
        denominator_tri_nonzero = torch.sign(edges_to_interp_msdf_tri[:,:,0]).sum(dim=1).abs() != 2
        denominator_quad_nonzero = torch.sign(edges_to_interp_msdf_quad[:,:,0]).sum(dim=1).abs() != 2

        edges_to_interp_msdf_tri[:,-1] *= -1
        edges_to_interp_msdf_quad[:,-1] *= -1
        denominator_tri = edges_to_interp_msdf_tri.sum(1, keepdim=True)
        denominator_quad = edges_to_interp_msdf_quad.sum(1, keepdim=True)

        denominator_tri_nonzero = (denominator_tri[:,0,0].abs() > 1e-12) & denominator_tri_nonzero
        denominator_quad_nonzero = (denominator_quad[:,0,0].abs() > 1e-12) & denominator_quad_nonzero

        edges_to_interp_msdf_tri_new = torch.zeros_like(edges_to_interp_msdf_tri)
        edges_to_interp_msdf_quad_new = torch.zeros_like(edges_to_interp_msdf_quad)
        edges_to_interp_msdf_tri_new[denominator_tri_nonzero] = torch.flip(edges_to_interp_msdf_tri[denominator_tri_nonzero], [1]) / denominator_tri[denominator_tri_nonzero]
        edges_to_interp_msdf_quad_new[denominator_quad_nonzero] = torch.flip(edges_to_interp_msdf_quad[denominator_quad_nonzero], [1]) / denominator_quad[denominator_quad_nonzero]

        edges_to_interp_msdf_tri = edges_to_interp_msdf_tri_new
        edges_to_interp_msdf_quad = edges_to_interp_msdf_quad_new

        ### Append additional boundary vertices (with negligible corner cases). Notice that unused vertices are included for efficiency reasons.
        verts_aug = torch.cat([
                    verts,
                    (edges_to_interp_vpos_tri * edges_to_interp_msdf_tri).sum(1), 
                    (edges_to_interp_vpos_quad * edges_to_interp_msdf_quad).sum(1)
                ],
            dim=0)

        ### NOTE: important to stop gradients from passing through the 'interpolation coefficients' (basically the 'coordinates' of boundary vertices)
        msdf_vert_tri_stopvgd = (edges_to_interp_msdf_tri_stopvgd * edges_to_interp_msdf_tri.detach()).sum(1).squeeze(dim=-1)
        msdf_vert_quad_stopvgd = (edges_to_interp_msdf_quad_stopvgd * edges_to_interp_msdf_quad.detach()).sum(1).squeeze(dim=-1)

        msdf_vert_aug_stopvgd = torch.cat([
            msdf_vert_stopvgd,
            msdf_vert_tri_stopvgd,
            msdf_vert_quad_stopvgd,
        ])

        msdf_vert_boundary_stopvgd = msdf_vert_aug_stopvgd[msdf_vert.size(0):] ## not all boundary vertices but good enough

        ### Determine how to cut polygon faces by checking the look-up tables
        with torch.no_grad():
            v_id_msdf_tri = torch.flip(torch.pow(2, torch.arange(3, dtype=torch.long, device=pos_nx3.device)), dims=[0])
            v_id_msdf_quad = torch.flip(torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device)), dims=[0])
            mesh_index_tri = (mocc_fx3 * v_id_msdf_tri.unsqueeze(0)).sum(-1)
            mesh_index_quad = (mocc_fx4 * v_id_msdf_quad.unsqueeze(0)).sum(-1)


        idx_map_tri = torch.cat([mesh_edge_tri[:, :, 0], verts.size(0) + torch.arange(mesh_edge_tri.size(0) * 3, device=pos_nx3.device).view(-1, 3)], dim=-1)
        idx_map_quad = torch.cat([mesh_edge_quad[:, :, 0], verts.size(0) + mesh_edge_tri.size(0) * 3 + torch.arange(mesh_edge_quad.size(0) * 4, device=pos_nx3.device).view(-1, 4)], dim=-1)

        num_triangles_tri = self.num_triangles_tri_table[mesh_index_tri]
        num_triangles_quad = self.num_triangles_quad_table[mesh_index_quad]

        ### Cut the polygon faces (case-by-case)
        faces_aug = torch.cat((
            torch.gather(input=idx_map_tri[num_triangles_tri == 1], dim=1, index=self.triangle_table_tri[mesh_index_tri[num_triangles_tri == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_tri[num_triangles_tri == 2], dim=1, index=self.triangle_table_tri[mesh_index_tri[num_triangles_tri == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 1], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 1]][:, :3]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 2], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 2]][:, :6]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 3], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 3]][:, :9]).view(-1, 3),
            torch.gather(input=idx_map_quad[num_triangles_quad == 4], dim=1, index=self.triangle_table_quad[mesh_index_quad[num_triangles_quad == 4]][:, :12]).view(-1, 3),
        ), dim=0)

        ### Mark all unused vertices (only for convenience in visualization; not necessary)
        with torch.no_grad():
            referenced_vert_idx = faces_aug.unique()
            mask = torch.ones(verts_aug.size(0))
            mask[referenced_vert_idx] = 0
        verts_aug[mask.bool()] = 0

        if output_watertight_template:
            extra = {
                'n_verts_watertight': verts.size(0),
                'vertices_watertight': verts,
                'faces_watertight': faces, 
                'msdf': msdf_vert_aug_stopvgd,
                'msdf_watertight': msdf_vert_stopvgd,
                'msdf_boundary': msdf_vert_boundary_stopvgd,
            }
        else:
            extra = {
                'msdf': msdf_vert_aug_stopvgd,
                'msdf_watertight': msdf_vert_stopvgd,
                'msdf_boundary': msdf_vert_boundary_stopvgd,
            }

        return verts_aug, faces_aug, extra

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )

        return mesh

    def forward_gshell(
        self,
        sdf: Float[Tensor, "N3 1"],
        msdf: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
        output_watertight_template_flag = True
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx, extra = self._forward_gshell(grid_vertices, sdf, msdf, self.indices, output_watertight_template_flag)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_sdf=sdf,
            grid_msdf=msdf,
            grid_deformation=deformation,
        )

        if output_watertight_template_flag:
            extra["mesh_watertight"] =  Mesh(
                v_pos=extra["vertices_watertight"],
                t_pos_idx=extra["faces_watertight"]
            )
            extra.pop("vertices_watertight")
            extra.pop("faces_watertight")
        return mesh, extra