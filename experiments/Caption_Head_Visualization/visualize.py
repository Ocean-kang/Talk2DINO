"""Pure visualization: no plotting dependencies beyond NumPy/Pillow/Matplotlib."""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from matplotlib import colormaps


BACKGROUND = '#ffffff'
INK = '#22272d'
MUTED = '#526170'


def font(size=18):
    for p in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit_caption(text, width_chars=43, max_lines=4):
    lines = textwrap.wrap(str(text), width=width_chars, break_long_words=False) or ['']
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(0, width_chars - 3)] + '...'
    return '\n'.join(lines)


def to_heatmap(attention, target_hw, low=None, high=None):
    """Return display scalar [0,1], interpolating the raw attention first."""
    a = torch.as_tensor(attention).float().reshape(1, 1, *attention.shape[-2:])
    up = F.interpolate(a, size=target_hw, mode='bilinear', align_corners=False)[0, 0].numpy()
    if low is None:
        low = float(np.percentile(up, 5))
    if high is None:
        high = float(np.percentile(up, 99))
    if high - low < 1e-12:
        return np.zeros_like(up)
    return np.clip((up - low) / (high - low), 0, 1)


def overlay_attention(crop: Image.Image, attention, *, alpha=.65, global_range=None):
    arr = np.asarray(crop.convert('RGB'), dtype=np.float32)
    r = global_range or (None, None)
    heat = to_heatmap(attention, (crop.height, crop.width), low=r[0], high=r[1])
    color = colormaps.get_cmap('turbo')(heat)[..., :3] * 255.0
    weight = (alpha * heat)[..., None]
    result = (1 - weight) * arr + weight * color
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), 'RGB')


def shared_limits(attentions):
    # Spatial interpolation is linear. Approximate the global display limits
    # directly on low resolution maps for speed; every head gets the same scale.
    a = torch.as_tensor(attentions).float().numpy()
    return float(np.percentile(a, 5)), float(np.percentile(a, 99))


def save_similarity_matrix(similarity, out_path):
    sims = np.asarray(similarity, dtype=np.float32)
    rows, cols = sims.shape
    cell_w, cell_h, left, top = 74, 48, 105, 57
    im = Image.new('RGB', (left + cols * cell_w + 20, top + rows * cell_h + 37), 'white')
    draw = ImageDraw.Draw(im)
    title_font, f = font(21), font(14)
    draw.text((18, 10), 'Caption-to-Head cosine similarity', fill=INK, font=title_font)
    for h in range(cols):
        draw.text((left + h * cell_w + 20, top - 25), f'H{h + 1}', fill=INK, font=f)
    # All cells of an image share one comparable color scale.
    low, high = float(sims.min()), float(sims.max())
    if high <= low:
        high = low + 1
    cmap = colormaps.get_cmap('viridis')
    for i in range(rows):
        draw.text((20, top + i * cell_h + 12), f'Caption {i + 1}', fill=INK, font=f)
        for h in range(cols):
            x, y = left + h * cell_w, top + i * cell_h
            val = (sims[i, h] - low) / (high - low)
            col = tuple(int(c * 255) for c in cmap(float(val))[:3])
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=col)
            draw.text((x + 14, y + 14), f'{sims[i, h]:.3f}', fill='white' if val < .6 else 'black', font=f)
        imax = int(sims[i].argmax())
        x = left + imax * cell_w
        y = top + i * cell_h
        draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), outline='red', width=3)
    draw.text((18, top + rows * cell_h + 6), f'Range: {low:.3f} to {high:.3f} (per image)', fill=MUTED, font=f)
    im.save(out_path)


def make_panel(image: Image.Image, label: str, caption='', side=320):
    '''Single caption card, same size for all panels.'''
    im = image.convert('RGB').resize((side, side), Image.Resampling.BILINEAR)
    panel = Image.new('RGB', (side + 20, side + 125), BACKGROUND)
    panel.paste(im, (10, 10))
    d = ImageDraw.Draw(panel)
    d.text((10, side + 17), label, fill=INK, font=font(17))
    d.multiline_text((10, side + 47), fit_caption(caption, width_chars=max(20, side // 9), max_lines=3),
                     fill=MUTED, font=font(14), spacing=4)
    return panel


def grid_cards(cards, out_path, columns=3, *, title=None):
    if not cards:
        raise ValueError('No cards')
    w, h = cards[0].size
    nrows = math.ceil(len(cards) / columns)
    header = 45 if title else 0
    grid = Image.new('RGB', (columns * w, nrows * h + header), '#eef2f7')
    if title:
        ImageDraw.Draw(grid).text((14, 9), title, fill=INK, font=font(22))
    for i, panel in enumerate(cards):
        grid.paste(panel, ((i % columns) * w, (i // columns) * h + header))
    grid.save(out_path)


def save_visualizations(crop, attentions, scores, descriptions, out_dir, *, alpha=.65, norm_mode='per-head', all_heads=True, full_resized=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir / 'original_crop.png')
    if full_resized is not None:
        full_resized.save(out_dir / 'original_full_view.png')
    attn = torch.as_tensor(attentions).float()
    if attn.ndim != 3 or attn.shape[0] != scores['similarities'].shape[1]:
        raise ValueError('Attention tensor head count mismatch')
    shared = shared_limits(attn) if norm_mode == 'global' else None
    image_cards = [make_panel(crop, 'DINO input: center crop', 'Resize shorter side to 448 and center crop.')]
    for row in descriptions['captions']:
        h = row['head_index']
        overlay = overlay_attention(crop, attn[h], alpha=alpha, global_range=shared)
        filename = f'caption_{row["caption_index"]:02d}_head_{h + 1:02d}.png'
        overlay.save(out_dir / filename)
        row['heatmap_file'] = filename
        if full_resized is not None:
            # Overlay only inside the original DINO crop; unseen context is unchanged.
            full = full_resized.copy()
            left = int(round((full.width - crop.width) / 2.0))
            top = int(round((full.height - crop.height) / 2.0))
            full.paste(overlay, (left, top))
            full_filename = filename.replace('.png', '_full_view.png')
            full.save(out_dir / full_filename)
            row['full_view_heatmap_file'] = full_filename
        image_cards.append(make_panel(overlay, f'Caption {row["caption_index"]+1} => Head {h+1}   cos={row["cosine_similarity"]:.3f}', row['caption']))
    grid_cards(image_cards, out_dir / 'overview.png', 3,
               title=f'Caption -> Head | view: {norm_mode} normalized attention')
    save_similarity_matrix(scores['similarities'].numpy(), out_dir / 'similarity_matrix.png')
    if all_heads:
        cards = []
        for h in range(attn.shape[0]):
            n = descriptions['head_use_counts'][h]
            overlay = overlay_attention(crop, attn[h], alpha=alpha, global_range=shared)
            cards.append(make_panel(overlay, f'Head {h+1} | matched captions: {n}', 'CLS-to-patch spatial attention'))
        grid_cards(cards, out_dir / 'all_12_heads.png', 4, title=f'All heads | {norm_mode} normalized attention')
    # Raw maps always stored for quantitative reuse; PNGs are display-normalized.
    np.savez_compressed(out_dir / 'attention_maps.npz', attention=attn.numpy().astype(np.float32))
