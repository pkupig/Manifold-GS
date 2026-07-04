"""Export a traceable hybrid asset bundle from certified manifold patches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mesh_io import read_triangle_mesh_ply, write_grouped_obj, write_triangle_mesh_ply
from .ply_io import read_vertex_ply, write_vertex_ply_data


def mesh_topology_statistics(faces: np.ndarray) -> dict[str, int | float]:
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        return {"boundary_edges": 0, "nonmanifold_edges": 0, "nonmanifold_edge_ratio": 0.0}
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "nonmanifold_edge_ratio": float(np.mean(counts > 2)),
    }


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices unused by faces and remap triangle indices."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.float32), faces
    used = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return np.asarray(vertices, dtype=np.float32)[used], remap[faces]


def export_asset_bundle(
    gaussian_ply: str | Path,
    patch_mesh_ply: str | Path,
    patch_meta_npz: str | Path,
    output_dir: str | Path,
    collision_min_faces: int = 8,
    gaussian_source_map_npz: str | Path | None = None,
    collision_max_patch_diameter_ratio: float = 3.0,
) -> dict[str, object]:
    """Write mesh, attached/residual splats, mappings, and a JSON manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = read_triangle_mesh_ply(patch_mesh_ply)
    meta = np.load(patch_meta_npz)
    source_indices = np.asarray(meta["source_indices"], dtype=np.int64).reshape(-1)
    patch_ids = np.asarray(meta["patch_ids"], dtype=np.int32).reshape(-1)
    if source_indices.shape[0] != vertices.shape[0] or patch_ids.shape[0] != vertices.shape[0]:
        raise ValueError("patch metadata must have one source index and patch id per mesh vertex")

    gaussian_data = read_vertex_ply(gaussian_ply).data
    if gaussian_source_map_npz is None:
        gaussian_source_ids = np.arange(len(gaussian_data), dtype=np.int64)
    else:
        source_map = np.load(gaussian_source_map_npz)
        gaussian_source_ids = np.asarray(source_map["source_indices"], dtype=np.int64).reshape(-1)
        if gaussian_source_ids.size != len(gaussian_data):
            raise ValueError("Gaussian source map must have one source index per PLY row")
    row_by_source = {int(source_id): row for row, source_id in enumerate(gaussian_source_ids)}
    missing_sources = sorted(set(map(int, source_indices)) - set(row_by_source))
    if missing_sources:
        raise ValueError(f"mesh refers to {len(missing_sources)} Gaussian IDs absent from the source map")
    attached_indices = np.unique([row_by_source[int(source_id)] for source_id in source_indices])
    residual_mask = np.ones(len(gaussian_data), dtype=bool)
    residual_mask[attached_indices] = False
    residual_indices = np.flatnonzero(residual_mask)

    write_vertex_ply_data(output_dir / "attached_gaussians.ply", gaussian_data[attached_indices])
    write_vertex_ply_data(output_dir / "residual_gaussians.ply", gaussian_data[residual_indices])
    write_triangle_mesh_ply(output_dir / "certified_patches.ply", vertices, faces)
    write_grouped_obj(output_dir / "certified_patches.obj", vertices, faces, patch_ids)

    # A conservative collision candidate keeps only sufficiently supported patches.
    face_patch = np.asarray(
        [int(np.bincount(patch_ids[tri].clip(min=0)).argmax()) for tri in faces], dtype=np.int32
    ) if faces.size else np.empty((0,), dtype=np.int32)
    unique_patch, patch_face_counts = np.unique(face_patch, return_counts=True)
    patch_diameters = np.asarray([
        float(np.linalg.norm(np.ptp(vertices[patch_ids == patch_id], axis=0)))
        for patch_id in unique_patch
    ])
    median_diameter = float(np.median(patch_diameters)) if patch_diameters.size else 0.0
    supported = patch_face_counts >= collision_min_faces
    scale_valid = patch_diameters <= collision_max_patch_diameter_ratio * max(median_diameter, 1e-12)
    retained_patch = unique_patch[supported & scale_valid]
    rejected_scale_patch = unique_patch[supported & ~scale_valid]
    collision_faces = faces[np.isin(face_patch, retained_patch)]
    collision_vertices, collision_faces = compact_mesh(vertices, collision_faces)
    write_triangle_mesh_ply(output_dir / "collision_candidate.ply", collision_vertices, collision_faces)

    np.savez_compressed(
        output_dir / "asset_mapping.npz",
        source_indices=source_indices,
        patch_ids=patch_ids,
        attached_source_indices=gaussian_source_ids[attached_indices],
        residual_source_indices=gaussian_source_ids[residual_indices],
        attached_row_indices=attached_indices,
        residual_row_indices=residual_indices,
        collision_patch_ids=retained_patch,
        collision_scale_rejected_patch_ids=rejected_scale_patch,
    )
    topology = mesh_topology_statistics(faces)
    collision_topology = mesh_topology_statistics(collision_faces)
    manifest: dict[str, object] = {
        "schema": "manifoldgs.hybrid_asset.v1",
        "status": "asset_backbone_not_production_ready",
        "source_gaussians": int(len(gaussian_data)),
        "source_id_mapping": "explicit" if gaussian_source_map_npz is not None else "identity",
        "attached_gaussians": int(attached_indices.size),
        "residual_gaussians": int(residual_indices.size),
        "attachment_ratio": float(attached_indices.size / max(len(gaussian_data), 1)),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
        "patches": int(np.unique(patch_ids).size),
        "collision_faces": int(collision_faces.shape[0]),
        "collision_vertices": int(collision_vertices.shape[0]),
        "collision_patches": int(retained_patch.size),
        "collision_scale_rejected_patches": [int(x) for x in rejected_scale_patch],
        "patch_diameter_median": median_diameter,
        **topology,
        "collision_nonmanifold_edges": collision_topology["nonmanifold_edges"],
        "files": {
            "mesh_obj": "certified_patches.obj",
            "mesh_mtl": "certified_patches.mtl",
            "mesh_ply": "certified_patches.ply",
            "attached_gaussians": "attached_gaussians.ply",
            "residual_gaussians": "residual_gaussians.ply",
            "collision_candidate": "collision_candidate.ply",
            "mapping": "asset_mapping.npz",
        },
    }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
