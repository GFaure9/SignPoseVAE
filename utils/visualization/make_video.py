import textwrap
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Any, Dict

DEFAULT_LIMS = (
    (-1.5, 1.5),
    (0, 2.),
    (-0.1, 0.1),
)


def make_video_simple(
    seq1: np.ndarray,
    bones: List[Tuple[int, int, Any]],
    no_link_ids: List[int],
    face_ids: List[int] = None,
    rh_ids: List[int] = None,
    lh_ids: List[int] = None,
    lims: List[Tuple] = None,
    text_str: str = None,
    score_str: str = None,
    out_file="test_mp128.mp4",
    fps=30,
    dpi=300,
    seq2: np.ndarray = None,  # optional
    text_str2: str = None,  # optional,
    label1: str = "Ground Truth", label2: str = "Prediction",
    joints_neighbourhood: Dict[int, float] = None,  # {ID of joint: radius of joint neighborhood to be plotted as a ball around it}  
):
    # Handle second sequence
    if seq2 is None:
        seq2 = np.zeros((0,))
    has_second = seq2.size > 0

    T1 = len(seq1)
    T2 = seq2.shape[0] if has_second else 0
    T = max(T1, T2)
    
    fig = plt.figure(figsize=(10 if has_second else 6, 6))

    def get_limits(seq, margin=0.05):
        pts = seq.reshape(-1, 3)
        mins, maxs = pts.min(0), pts.max(0)
        span = (maxs - mins) * margin
        return (mins[0] - span[0], maxs[0] + span[0]), \
               (mins[1] - span[1], maxs[1] + span[1]), \
               (mins[2] - span[2], maxs[2] + span[2])

    if has_second:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")  # left: Ground Truth
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")  # right: Prediction
    else:
        ax1 = fig.add_subplot(111, projection="3d")
        ax2 = None

    if lims is None:
        xlim1, ylim1, zlim1 = get_limits(seq1)
        if has_second:
            xlim2, ylim2, zlim2 = get_limits(seq2)
        else:
            xlim2 = ylim2 = zlim2 = None
    else:
        xlim1, ylim1, zlim1 = lims
        xlim2, ylim2, zlim2 = lims if has_second else (None, None, None)

    if joints_neighbourhood:
        U, V = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        X0 = np.cos(U) * np.sin(V)
        Y0 = np.sin(U) * np.sin(V)
        Z0 = np.cos(V)
            
    def draw_single(ax, frame, xlim, ylim, zlim, masks_colors, label=None, label_color="black", freeze=False):
        ax.cla()
        pts = frame.copy()
        for mask, color in masks_colors:
            col = "grey" if freeze else color
            for i, j, _ in bones:
                if (i not in no_link_ids) and mask[i] and mask[j]:
                    ax.plot(*pts[[i, j]].T, c=col, lw=0.7)
            ax.scatter(*pts[mask].T, c=col, s=5)
        if joints_neighbourhood:
            for i, pt in enumerate(pts):
                if i in joints_neighbourhood:
                    r = joints_neighbourhood[i]
                    ax.plot_surface(pt[0] + r*X0,
                                    pt[1] + r*Y0,
                                    pt[2] + r*Z0,
                                    color="red", alpha=0.3, linewidth=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=95, azim=-90)
        ax.set_xlabel("X", alpha=0.8)
        ax.set_ylabel("Y", alpha=0.8)
        ax.set_zlabel("Z", alpha=0.8)
        if label is not None:
            label_y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.2
            ax.text((xlim[0]+xlim[1])/2, label_y_pos, zlim[1], label,
                    ha="center", fontsize=12, weight="bold", color=label_color)

    # Define mask colors
    n_pts = len(seq1[0])
    mask_face = np.zeros(n_pts, bool)
    mask_rh = np.zeros(n_pts, bool)
    mask_lh = np.zeros(n_pts, bool)
    mask_rem = np.ones(n_pts, bool)

    masks_colors = []
    if face_ids is not None:
        mask_face[face_ids] = True
        mask_rem[face_ids] = False
        masks_colors.append((mask_face, "magenta"))
    if rh_ids is not None:
        mask_rh[rh_ids] = True
        mask_rem[rh_ids] = False
        masks_colors.append((mask_rh, "green"))
    if lh_ids is not None:
        mask_lh[lh_ids] = True
        mask_rem[lh_ids] = False
        masks_colors.append((mask_lh, "red"))
    masks_colors.append((mask_rem, "blue"))

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, out_file, dpi=dpi):
        for t in tqdm(range(T)):
            t1_done = t >= T1
            t2_done = t >= T2
            f1 = seq1[min(t, T1 - 1)]
            f2 = seq2[min(t, T2 - 1)] if has_second else None

            if has_second:
                draw_single(ax1, f1, xlim1, ylim1, zlim1, masks_colors, label=label1, freeze=t1_done, label_color="green")
                draw_single(ax2, f2, xlim2, ylim2, zlim2, masks_colors, label=label2, freeze=t2_done, label_color="blue")
                
                if text_str is not None:
                    wrapped_text = "\n".join(textwrap.wrap(text_str, width=40))
                    x = xlim1[0] + (xlim1[1] - xlim1[0]) * 0.5
                    y = ylim1[1] + (ylim1[1] - ylim1[0]) * 0.5
                    z = zlim1[1]
                    ax1.text(x, y, z, wrapped_text, ha="center", fontsize=10)
                if text_str2 is not None:
                    wrapped_text = "\n".join(textwrap.wrap(text_str2, width=40))
                    n_lines = wrapped_text.count("\n") + 1  # number of lines
                    x = xlim2[0] + (xlim2[1] - xlim2[0]) * 0.5
                    y = ylim2[1] + (ylim2[1] - ylim2[0]) * 0.5
                    z = zlim2[1]
                    ax2.text(x, y, z, wrapped_text, ha="center", fontsize=10, va="top")
                if score_str is not None:
                    line_spacing = (ylim2[1] - ylim2[0]) * 0.1
                    y_score = y - n_lines * line_spacing
                    ax2.text(x, y_score, z, score_str, ha="center", va="top", weight="bold")
                # if text_str2 is not None:
                #     wrapped_text = "\n".join(textwrap.wrap(text_str2, width=40))
                #     x = xlim2[0] + (xlim2[1] - xlim2[0]) * 0.5
                #     y = ylim2[1] + (ylim2[1] - ylim2[0]) * 0.5
                #     z = zlim2[1]
                #     ax2.text(x, y, z, wrapped_text, ha="center", fontsize=10)
                # if score_str is not None:
                #     x = xlim2[0] + (xlim2[1] - xlim2[0]) * 0.5
                #     y = ylim2[1] + (ylim2[1] - ylim2[0]) * 0.5
                #     z = zlim2[1]
                #     ax2.text(x, y, z, score_str, ha="center", weight="bold")
                
            else:
                draw_single(ax1, f1, xlim1, ylim1, zlim1, masks_colors, label=None, freeze=t1_done)
                title = ""
                if text_str is not None:
                    title += text_str
                if score_str is not None:
                    title += f"\n{score_str}"
                ax1.set_title(title, y=1.02)

            writer.grab_frame()

    plt.close(fig)


