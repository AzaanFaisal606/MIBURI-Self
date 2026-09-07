"""Blender-side helper: pull each submesh's diffuse texture out of an FBX.

    blender --background --python _blender_extract_textures.py -- <in.fbx> <out_dir>

Writes one PNG per mesh object, named after the object, plus an `index.txt`
listing `<object_name>\t<png_filename>` so the caller can pair textures with
the `submesh_names` recorded in the character bundle.

Mesh objects are visited in the same order `_blender_fbx_export.py` combines
them (sorted by name, empty meshes skipped), so the ordering matches
`submesh_ranges`.
"""

import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
fbx_path, out_dir = argv[0], argv[1]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)


def base_color_image(material):
    """Follow the material's Base Color input back to an image node."""
    if material is None or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        link = node.inputs["Base Color"].links
        if not link:
            continue
        source = link[0].from_node
        # Base Color may go through a mix/gamma node; walk one hop back.
        if source.type == "TEX_IMAGE":
            return source.image
        for sub in source.inputs:
            for sublink in sub.links:
                if sublink.from_node.type == "TEX_IMAGE":
                    return sublink.from_node.image
    # Fall back to any image node in the tree.
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image
    return None


meshes = sorted(
    (o for o in bpy.data.objects if o.type == "MESH" and len(o.data.vertices) > 0),
    key=lambda o: o.name,
)

entries = []
for obj in meshes:
    image = None
    for slot in obj.material_slots:
        image = base_color_image(slot.material)
        if image is not None:
            break
    if image is None:
        print(f"[textures] {obj.name}: no diffuse image")
        entries.append((obj.name, ""))
        continue

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name)
    png_path = os.path.join(out_dir, f"{safe}.png")
    image.file_format = "PNG"
    # Packed FBX textures have no filepath on disk; save_render works either way.
    image.save_render(filepath=png_path)
    print(f"[textures] {obj.name}: {image.size[0]}x{image.size[1]} -> {png_path}")
    entries.append((obj.name, os.path.basename(png_path)))

with open(os.path.join(out_dir, "index.txt"), "w") as handle:
    for name, png in entries:
        handle.write(f"{name}\t{png}\n")

print(f"[textures] wrote {len(entries)} entries")
