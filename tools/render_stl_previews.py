"""Render binary or ASCII STL meshes to orthographic PNG previews (NumPy, Pillow).

Run from the repository root: python tools/render_stl_previews.py
"""
from pathlib import Path
import re
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'media' / 'stl-previews'


def load_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + count * 50:
        dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        return np.frombuffer(data, dtype=dtype, count=count, offset=84)['vertices'].astype(float)
    vertices = re.findall(rb'vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)', data)
    return np.array(vertices, dtype=float).reshape(-1, 3, 3)


def render(path):
    mesh = load_stl(path)
    low, high = mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))
    mesh -= (low + high) / 2
    # Fixed isometric camera; each part fits the same image area independently.
    view = np.array([1.0, -1.0, 0.85])
    view /= np.linalg.norm(view)
    right = np.cross([0, 0, 1], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    projected = mesh @ np.stack([right, up, view], axis=1)
    size, supersample = 480, 3
    canvas = Image.new('RGB', (size * supersample, size * supersample), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    bounds_min = projected[:, :, :2].min(axis=(0, 1))
    bounds_max = projected[:, :, :2].max(axis=(0, 1))
    scale = 370 / max(bounds_max - bounds_min)
    center = (bounds_min + bounds_max) / 2
    xy = (projected[:, :, :2] - center) * [scale, -scale] + [240, 218]
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    light = np.array([-0.4, -0.6, 1.0])
    light /= np.linalg.norm(light)
    # Two-sided shading accommodates STL files with inconsistent winding.
    normals *= np.where((normals @ view)[:, None] < 0, -1, 1)
    brightness = 0.48 + 0.52 * np.maximum(0, normals @ light)
    base = np.array([220, 68, 48])
    for index in np.argsort(projected[:, :, 2].mean(axis=1)):
        if lengths[index] < 1e-10:
            continue
        color = tuple((base * brightness[index]).astype(int))
        points = [tuple(point * supersample) for point in xy[index]]
        draw.polygon(points, fill=color)
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    draw.text((240, 448), path.name, fill='#445163', anchor='mm', font=ImageFont.load_default(size=18))
    canvas.save(OUT / (path.stem + '.png'))
    print(f'{path.name}: {len(mesh)} triangles')
    return canvas


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / 'stl').glob('*.stl'))
    images = [render(path) for path in paths]
    sheet = Image.new('RGB', (480 * 4, 480 * 3), 'white')
    for index, preview in enumerate(images):
        sheet.paste(preview, ((index % 4) * 480, (index // 4) * 480))
    sheet.save(OUT / 'overview.png')
