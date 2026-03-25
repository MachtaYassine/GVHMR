"""
nvdiffrast-based renderer — drop-in replacement for Renderer (PyTorch3D).

Uses NVIDIA nvdiffrast CUDA backend for rasterization + interpolation,
with manual Lambertian shading. All operations stay GPU-resident.
"""

import cv2
import torch
import numpy as np
import nvdiffrast.torch as dr

from .renderer_tools import get_colors, checkerboard_geometry


class NvDiffRastRenderer:
    """Drop-in replacement for Renderer using nvdiffrast CUDA backend."""

    def __init__(self, width, height, focal_length=None, device="cuda", faces=None, K=None, render_scale=1.0):
        self.width = width
        self.height = height
        self.device = device
        self.render_scale = render_scale
        self.rw = int(width * render_scale)
        self.rh = int(height * render_scale)

        # Ensure CUDA is initialized before creating nvdiffrast context
        torch.cuda.init()
        torch.cuda.empty_cache()

        # Persistent GL context — created once, reused for all frames
        self.glctx = dr.RasterizeCudaContext(device=device)

        # Faces (int32 required by nvdiffrast)
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy(faces.astype(np.int32))
            self.faces = faces.to(torch.int32).to(device)
        else:
            self.faces = None

        # Intrinsics
        assert (focal_length is not None) ^ (K is not None), "focal_length and K are mutually exclusive"
        if K is not None:
            if isinstance(K, np.ndarray):
                K = torch.from_numpy(K)
            self.K = K.float().reshape(3, 3).to(device)
        else:
            self.K = torch.tensor([
                [focal_length, 0, width / 2.0],
                [0, focal_length, height / 2.0],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

        # Build projection matrix once
        self._build_proj_matrix()

        # Light direction in OpenCV camera space (camera looks along +z)
        self.light_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        self.ambient = 0.55
        self.diffuse = 0.45

    def _build_proj_matrix(self):
        """Build projection matrix for nvdiffrast from OpenCV pinhole intrinsics.

        OpenCV camera: x-right, y-down, z-forward (positive z in front).
        We convert to OpenGL clip space by flipping y and z in the matrix itself,
        so _to_clip only needs to pass vertices through unchanged.
        """
        fx = self.K[0, 0] * self.render_scale
        fy = self.K[1, 1] * self.render_scale
        cx = self.K[0, 2] * self.render_scale
        cy = self.K[1, 2] * self.render_scale
        w, h = self.rw, self.rh
        near, far = 0.01, 100.0

        # Standard OpenGL projection but with OpenCV->OpenGL sign flips baked in:
        # - Row 1 (y): negated to flip y-down -> y-up
        # - Row 2,3 (z,w): negated to flip z-forward -> z-backward
        self.proj = torch.zeros(4, 4, device=self.device, dtype=torch.float32)
        self.proj[0, 0] = 2.0 * fx / w
        self.proj[0, 2] = (2.0 * cx - w) / w
        self.proj[1, 1] = -(2.0 * fy / h)                  # flip y (y-down -> y-up)
        self.proj[1, 2] = -(2.0 * cy - h) / h              # flip y
        self.proj[2, 2] = (far + near) / (far - near)       # map z to NDC
        self.proj[2, 3] = -2.0 * far * near / (far - near)  # z offset (negative!)
        self.proj[3, 2] = 1.0                                # w = z (positive z in front)

    def _to_clip(self, verts):
        """Transform camera-space vertices to clip coordinates.

        Accepts (V, 3) -> (1, V, 4) or (B, V, 3) -> (B, V, 4).
        The OpenCV->OpenGL conversion (y/z flip) is baked into the projection matrix.
        """
        if verts.dim() == 2:
            V = verts.shape[0]
            v4 = torch.ones(V, 4, device=self.device, dtype=torch.float32)
            v4[:, :3] = verts
            clip = v4 @ self.proj.T
            return clip.unsqueeze(0)  # (1, V, 4)
        else:
            B, V, _ = verts.shape
            v4 = torch.ones(B, V, 4, device=self.device, dtype=torch.float32)
            v4[:, :, :3] = verts
            clip = v4 @ self.proj.T  # (B, V, 4)
            return clip

    def _compute_face_normals(self, verts, faces):
        """Compute per-face normals from vertices."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
        return normals

    def render_mesh(self, vertices, background=None, colors=[0.8, 0.8, 0.8], VI=50):
        """Render mesh overlay on background image. Same interface as Renderer.render_mesh."""
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).float().to(self.device)
        if vertices.dim() == 2:
            verts = vertices  # (V, 3)
        else:
            verts = vertices.squeeze(0)

        faces = self.faces

        # Clip-space transform
        clip_verts = self._to_clip(verts)

        # Rasterize
        rast_out, _ = dr.rasterize(self.glctx, clip_verts, faces, resolution=[self.rh, self.rw])

        # Per-vertex colors
        if isinstance(colors, torch.Tensor) and colors.dim() >= 2:
            # Per-vertex color tensor
            vert_colors = colors.squeeze(0).to(device=self.device, dtype=torch.float32)
            if vert_colors.shape[-1] > 3:
                vert_colors = vert_colors[..., :3]
        else:
            if isinstance(colors, (list, tuple)):
                if colors[0] > 1:
                    colors = [c / 255.0 for c in colors]
            vert_colors = torch.tensor(colors, device=self.device, dtype=torch.float32).unsqueeze(0).expand(verts.shape[0], -1)

        # Interpolate colors
        color_out, _ = dr.interpolate(vert_colors.unsqueeze(0).contiguous(), rast_out, faces)

        # Compute normals in camera space (two-sided lighting via abs handles winding)
        face_normals = self._compute_face_normals(verts, faces)
        # Assign face normal to each vertex of that face (flat shading via per-face)
        # We'll use the rast triangle_id to look up face normals
        tri_id = rast_out[0, :, :, 3:4].long() - 1  # (H, W, 1), 0-indexed (-1 for background)
        valid = tri_id >= 0

        # Two-sided Lambertian: use abs(cos) so back-faces also get lit
        cos_angle = torch.sum(face_normals * self.light_dir.unsqueeze(0), dim=1).abs()  # (F,)
        shade_per_face = self.ambient + self.diffuse * cos_angle  # (F,)

        # Build shade map
        shade_map = torch.zeros(self.rh, self.rw, 1, device=self.device, dtype=torch.float32)
        tri_id_flat = tri_id.squeeze(-1)  # (H, W)
        valid_flat = valid.squeeze(-1)    # (H, W)
        shade_map[valid_flat] = shade_per_face[tri_id_flat[valid_flat]].unsqueeze(-1)

        # Apply shading
        shaded = (color_out[0] * shade_map).clamp(0, 1)

        # Mask from rasterization (triangle_id > 0 means covered)
        mask = (rast_out[0, :, :, 3] > 0)

        # Flip vertically on GPU (OpenGL bottom-left -> top-left origin), then to numpy
        image = (shaded.flip(0) * 255).byte().cpu().numpy()
        mask_np = mask.flip(0).cpu().numpy()

        # Upscale if needed
        if self.render_scale != 1.0:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            mask_np = cv2.resize(mask_np.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST).astype(bool)

        if background is None:
            background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        out = background.copy()
        out[mask_np] = image[mask_np]
        return out

    def render_with_ground(self, verts, colors, cameras, lights, faces=None):
        """Render multiple meshes with ground plane. Same interface as Renderer.render_with_ground.

        :param verts (N, V, 3)
        :param colors (N, 3) or (N, V, 3)
        :param cameras: PyTorch3D PerspectiveCameras (used for R, T extraction)
        :param lights: PyTorch3D PointLights (used for light position)
        :param faces (N, F, 3) optional
        """
        N, V, _ = verts.shape

        if faces is None:
            mesh_faces = self.faces.unsqueeze(0).expand(N, -1, -1)
        else:
            mesh_faces = faces

        # Handle colors
        if len(colors.shape) == 2:
            colors = colors[:, None].expand(N, V, -1)[..., :3]
        else:
            colors = colors[..., :3].expand(N, V, -1)

        # Get ground geometry
        gv, gf, gc = self.ground_geometry
        gc = gc[..., :3]

        # Get camera R, T from PyTorch3D cameras
        # PyTorch3D convention: v_cam = v_world @ R + T (row-vector, R is stored as R.mT)
        # Our convention: v_cam = R_col @ v_world + T (column-vector)
        # PyTorch3D also uses NDC with y-up, z-away-from-camera
        # Our projection uses OpenCV (y-down, z-towards-camera) with flips baked in
        # Need to flip Y and Z of the camera-space result to go from PT3D -> OpenCV
        R = cameras.R[0].mT.to(self.device)  # undo the mT storage -> column-vector R
        T = cameras.T[0].to(self.device)
        # PT3D camera space: x-right, y-up, z-away -> OpenCV: x-right, y-down, z-towards
        flip_yz = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                               dtype=torch.float32, device=self.device)
        R = flip_yz @ R
        T = flip_yz @ T

        # Merge all meshes + ground into single vertex/face buffers
        all_verts = []
        all_faces = []
        all_colors = []
        vert_offset = 0

        for i in range(N):
            v = verts[i]  # (V, 3)
            f = mesh_faces[i]  # (F, 3)
            c = colors[i]  # (V, 3)
            all_verts.append(v)
            all_faces.append(f + vert_offset)
            all_colors.append(c)
            vert_offset += v.shape[0]

        # Ground
        all_verts.append(gv.to(self.device))
        all_faces.append(gf.to(self.device).to(torch.int32) + vert_offset)
        all_colors.append(gc.to(self.device))

        merged_verts = torch.cat(all_verts, dim=0).float()  # (total_V, 3)
        merged_faces = torch.cat(all_faces, dim=0).to(torch.int32)  # (total_F, 3)
        merged_colors = torch.cat(all_colors, dim=0).float()  # (total_V, 3)

        # Transform to camera space: v_cam = R @ v + T
        v_cam = (R @ merged_verts.T).T + T

        # Clip transform
        clip_verts = self._to_clip(v_cam)

        # Rasterize
        rast_out, _ = dr.rasterize(self.glctx, clip_verts, merged_faces, resolution=[self.rh, self.rw])

        # Interpolate colors
        color_out, _ = dr.interpolate(merged_colors.unsqueeze(0).contiguous(), rast_out, merged_faces)

        # Shading (two-sided Lambertian handles winding via abs)
        face_normals = self._compute_face_normals(v_cam, merged_faces)
        cos_angle = torch.sum(face_normals * self.light_dir.unsqueeze(0), dim=1).abs()
        shade_per_face = self.ambient + self.diffuse * cos_angle

        tri_id = rast_out[0, :, :, 3:4].long() - 1
        valid = (tri_id >= 0).squeeze(-1)
        tri_id_flat = tri_id.squeeze(-1)

        shade_map = torch.zeros(self.rh, self.rw, 1, device=self.device, dtype=torch.float32)
        shade_map[valid] = shade_per_face[tri_id_flat[valid]].unsqueeze(-1)

        shaded = (color_out[0] * shade_map).clamp(0, 1)
        image = (shaded.flip(0) * 255).byte().cpu().numpy()  # flip for top-left origin

        if self.render_scale != 1.0:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return image

    def render_triview(self, world_verts, world_faces, world_colors,
                       R_views, T_views, world_light_dir=None):
        """Render the same scene from multiple cameras in a single batched rasterize call.

        All geometry (body, ground, hands) is pre-merged by the caller.
        Shading is computed once in world space (Option B: view-independent).

        :param world_verts:  (V, 3) merged vertices in world/scene space
        :param world_faces:  (F, 3) int32 merged face indices
        :param world_colors: (V, 3) per-vertex RGB colors
        :param R_views:      (B, 3, 3) rotation matrices for each view
        :param T_views:      (B, 3) translation vectors for each view
        :param world_light_dir: (3,) light direction in world space, default: from above-front
        :return: list of B numpy images (H, W, 3) uint8
        """
        B = R_views.shape[0]
        V = world_verts.shape[0]

        # --- World-space shading (computed once, shared across views) ---
        if world_light_dir is None:
            # Light from above and slightly in front (y-up scene space)
            ld = torch.tensor([0.0, -0.8, 0.6], device=self.device, dtype=torch.float32)
            world_light_dir = ld / ld.norm()
        else:
            world_light_dir = world_light_dir.to(self.device)

        face_normals = self._compute_face_normals(world_verts, world_faces)  # (F, 3)
        cos_angle = torch.sum(face_normals * world_light_dir.unsqueeze(0), dim=1).abs()
        shade_per_face = self.ambient + self.diffuse * cos_angle  # (F,)

        # --- Batched camera transform ---
        # v_cam[b] = R[b] @ world_verts.T + T[b]  =>  (B, V, 3)
        v_cam = torch.bmm(R_views, world_verts.unsqueeze(0).expand(B, -1, -1).transpose(1, 2))
        v_cam = v_cam.transpose(1, 2) + T_views.unsqueeze(1)  # (B, V, 3)

        # --- Batched clip-space projection ---
        clip_verts = self._to_clip(v_cam)  # (B, V, 4)

        # --- Single batched rasterize ---
        rast_out, _ = dr.rasterize(self.glctx, clip_verts, world_faces,
                                   resolution=[self.rh, self.rw])  # (B, H, W, 4)

        # --- Batched color interpolation ---
        # Colors are the same for all views, expand to batch
        colors_batch = world_colors.unsqueeze(0).expand(B, -1, -1).contiguous()
        color_out, _ = dr.interpolate(colors_batch, rast_out, world_faces)  # (B, H, W, 3)

        # --- Build shade maps from shared shade_per_face ---
        tri_ids = rast_out[:, :, :, 3:4].long() - 1  # (B, H, W, 1)
        valid = (tri_ids >= 0)  # (B, H, W, 1)

        # Gather shade per pixel: shade_per_face[tri_id] for valid pixels
        tri_ids_clamped = tri_ids.clamp(min=0).squeeze(-1)  # (B, H, W)
        shade_gathered = shade_per_face[tri_ids_clamped]  # (B, H, W)
        shade_map = (shade_gathered * valid.squeeze(-1).float()).unsqueeze(-1)  # (B, H, W, 1)

        # --- Apply shading and produce images ---
        shaded = (color_out * shade_map).clamp(0, 1)  # (B, H, W, 3)
        images_gpu = (shaded.flip(1) * 255).byte()  # flip Y for all views at once

        # Transfer to CPU and split
        images_np = images_gpu.cpu().numpy()
        results = []
        for b in range(B):
            img = images_np[b]
            if self.render_scale != 1.0:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            results.append(img)

        return results

    # --- Compatibility methods ---
    def set_ground(self, length, center_x, center_z):
        device = self.device
        length, center_x, center_z = map(float, (length, center_x, center_z))
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def create_camera(self, R=None, T=None):
        """Compatibility: returns a PerspectiveCameras-like object for render_with_ground callers."""
        from pytorch3d.renderer import PerspectiveCameras
        if R is None:
            R = torch.eye(3, device=self.device).unsqueeze(0)
        else:
            R = R.clone().view(1, 3, 3).to(self.device)
        if T is None:
            T = torch.zeros(1, 3, device=self.device)
        else:
            T = T.clone().view(1, 3).to(self.device)
        return PerspectiveCameras(device=self.device, R=R.mT, T=T)

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """No-op for nvdiffrast (always renders full frame)."""
        pass

    def reset_bbox(self):
        """No-op for nvdiffrast."""
        pass

    def set_intrinsic(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.K = K.float().reshape(3, 3).to(self.device)
        self._build_proj_matrix()
