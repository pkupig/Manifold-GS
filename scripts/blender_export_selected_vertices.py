from pathlib import Path

import bpy
import bmesh

# Change only this value for the mesh currently open in Blender.
SCAN = 24
OUTPUT_DIR = Path("/root/autodl-tmp/E-Manifold-GS/annotations/external_regions")

obj = bpy.context.edit_object
if obj is None or obj.type != "MESH":
    raise RuntimeError("Enter Edit Mode on certified_patches.obj and select vertices first.")

bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()
selected = sorted(vertex.index for vertex in bm.verts if vertex.select)
if not selected:
    raise RuntimeError("No selected vertices. Select a region in Edit Mode first.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"scan{SCAN}_part_vertex_indices.txt"
output_path.write_text("\n".join(map(str, selected)) + "\n", encoding="utf-8")
print(f"Wrote {len(selected)} zero-based vertex IDs to {output_path}")
