"""Analytic surface scenes and a small deterministic CPU rasterizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class RenderedView:
    rgb: np.ndarray
    depth: np.ndarray
    normal_world: np.ndarray
    mask: np.ndarray


def look_at_w2c(eye: np.ndarray, target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(3, dtype=np.float64) if target is None else np.asarray(target, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    return rotation, -rotation @ eye


def create_scene_mesh(scene: str, resolution: int = 48) -> trimesh.Trimesh:
    if scene == "sphere":
        subdivisions = max(2, min(5, int(np.ceil(np.log2(max(resolution, 8) / 8)))))
        return trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.7)
    if scene == "torus":
        return trimesh.creation.torus(
            major_radius=0.52,
            minor_radius=0.20,
            major_sections=max(24, resolution),
            minor_sections=max(12, resolution // 2),
        )
    if scene == "plane":
        side = max(3, resolution)
        x = np.linspace(-0.75, 0.75, side)
        y = np.linspace(-0.75, 0.75, side)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
        faces: list[tuple[int, int, int]] = []
        for row in range(side - 1):
            for col in range(side - 1):
                a = row * side + col
                b = a + 1
                c = a + side
                d = c + 1
                faces.extend([(a, c, b), (b, c, d)])
        return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    raise ValueError(f"Unsupported analytic scene: {scene}")


def sample_analytic_surface(scene: str, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact surface points, unoriented normals, and quadrature weights."""
    rng = np.random.default_rng(seed)
    if scene == "sphere":
        z = rng.uniform(-1.0, 1.0, count)
        phi = rng.uniform(0.0, 2.0 * np.pi, count)
        radial = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        normals = np.stack([radial * np.cos(phi), radial * np.sin(phi), z], axis=1)
        points = 0.7 * normals
        area = 4.0 * np.pi * 0.7**2
    elif scene == "torus":
        # Rejection sampling gives uniform area because dA is proportional to R+r cos(v).
        major, minor = 0.52, 0.20
        us: list[np.ndarray] = []
        vs: list[np.ndarray] = []
        remaining = count
        while remaining:
            batch = max(remaining * 2, 128)
            u = rng.uniform(0.0, 2.0 * np.pi, batch)
            v = rng.uniform(0.0, 2.0 * np.pi, batch)
            accept = rng.random(batch) <= (major + minor * np.cos(v)) / (major + minor)
            us.append(u[accept][:remaining])
            vs.append(v[accept][:remaining])
            remaining -= min(remaining, int(np.sum(accept)))
        u = np.concatenate(us)[:count]
        v = np.concatenate(vs)[:count]
        normals = np.stack([np.cos(u) * np.cos(v), np.sin(u) * np.cos(v), np.sin(v)], axis=1)
        points = np.stack([
            (major + minor * np.cos(v)) * np.cos(u),
            (major + minor * np.cos(v)) * np.sin(u),
            minor * np.sin(v),
        ], axis=1)
        area = 4.0 * np.pi**2 * major * minor
    elif scene == "plane":
        points = np.column_stack([
            rng.uniform(-0.75, 0.75, count),
            rng.uniform(-0.75, 0.75, count),
            np.zeros(count),
        ])
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
        area = 1.5**2
    else:
        raise ValueError(f"Unsupported analytic scene: {scene}")
    weights = np.full(count, area / count, dtype=np.float64)
    return points.astype(np.float32), normals.astype(np.float32), weights.astype(np.float32)


def camera_eyes(scene: str, count: int) -> np.ndarray:
    if scene == "plane":
        azimuth = np.linspace(-0.42, 0.42, count)
        elevation = 0.18 * np.sin(np.arange(count) * 2.399963)
        return np.stack([2.4 * np.sin(azimuth), elevation, 2.4 * np.cos(azimuth)], axis=1)
    azimuth = np.arange(count) * (2.0 * np.pi / count)
    elevation = 0.34 * np.sin(np.arange(count) * 2.399963)
    radius = 2.4
    return np.stack([
        radius * np.cos(elevation) * np.cos(azimuth),
        radius * np.sin(elevation),
        radius * np.cos(elevation) * np.sin(azimuth),
    ], axis=1)


def surface_color(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    base = 0.28 + 0.48 * np.clip((points + 0.8) / 1.6, 0.0, 1.0)
    checker = ((np.floor((points[:, 0] + 1.0) * 5) + np.floor((points[:, 1] + 1.0) * 5)) % 2)[:, None]
    base *= 0.82 + 0.18 * checker
    light = np.array([0.35, 0.75, 0.56])
    light /= np.linalg.norm(light)
    diffuse = 0.55 + 0.45 * np.abs(normals @ light)
    return np.clip(base * diffuse[:, None], 0.0, 1.0)


def rasterize_mesh(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    translation: np.ndarray,
    width: int,
    height: int,
    focal: float,
) -> RenderedView:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    camera_vertices = vertices @ rotation.T + translation
    z = camera_vertices[:, 2]
    projected = np.column_stack([
        focal * camera_vertices[:, 0] / np.maximum(z, 1e-9) + width / 2.0,
        focal * camera_vertices[:, 1] / np.maximum(z, 1e-9) + height / 2.0,
    ])
    colors = surface_color(vertices, normals)

    depth = np.full((height, width), np.inf, dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    rgb[:] = np.array([0.035, 0.045, 0.06])
    normal_image = np.zeros((height, width, 3), dtype=np.float64)

    for face in np.asarray(mesh.faces):
        if np.any(z[face] <= 1e-6):
            continue
        p = projected[face]
        xmin = max(0, int(np.floor(np.min(p[:, 0]))))
        xmax = min(width - 1, int(np.ceil(np.max(p[:, 0]))))
        ymin = max(0, int(np.floor(np.min(p[:, 1]))))
        ymax = min(height - 1, int(np.ceil(np.max(p[:, 1]))))
        if xmin > xmax or ymin > ymax:
            continue
        denom = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(denom) < 1e-12:
            continue
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        px = xx + 0.5
        py = yy + 0.5
        b0 = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) + (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / denom
        b1 = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) + (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / denom
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-8) & (b1 >= -1e-8) & (b2 >= -1e-8)
        if not np.any(inside):
            continue
        bary = np.stack([b0, b1, b2], axis=-1)
        corrected = bary / z[face][None, None, :]
        corrected /= np.maximum(np.sum(corrected, axis=-1, keepdims=True), 1e-12)
        face_depth = np.sum(corrected * z[face][None, None, :], axis=-1)
        old = depth[ymin : ymax + 1, xmin : xmax + 1]
        update = inside & (face_depth < old)
        if not np.any(update):
            continue
        interpolated_color = corrected @ colors[face]
        interpolated_normal = corrected @ normals[face]
        interpolated_normal /= np.maximum(np.linalg.norm(interpolated_normal, axis=-1, keepdims=True), 1e-12)
        old[update] = face_depth[update]
        rgb_patch = rgb[ymin : ymax + 1, xmin : xmax + 1]
        normal_patch = normal_image[ymin : ymax + 1, xmin : xmax + 1]
        rgb_patch[update] = interpolated_color[update]
        normal_patch[update] = interpolated_normal[update]

    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return RenderedView(
        rgb=np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8),
        depth=depth.astype(np.float32),
        normal_world=normal_image.astype(np.float32),
        mask=mask,
    )

