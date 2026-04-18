"""Compute reprojection-error metrics from a calibration_detections.npz file.

Outputs a multi-panel PNG with three rows:

- Row 1: image-space (spatial) mean |residual| and observation count.
- Row 2: pose-stage mean |residual| and observation count (3×6 grid).
- Row 3: per-stage spatial heatmaps (3×6 tiling of spatial heatmaps), once
  for mean |residual| and once for observation count.

Optionally overwrites calibration.yaml with the stored calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, SubplotSpec

from calibrate_april_tag import (
    CALIBRATION_DETECTIONS_PATH,
    STAGE_DISTANCE_RANGES_M,
    STAGE_ORIENTATIONS,
    STAGES,
    is_frame_in_stage,
)
from calibration_types import (
    CalibrationResult,
    CameraMatrix,
    DistortionCoefficients,
    ImageSize,
)
from constants import CALIBRATION_YAML_PATH

SPATIAL_CELL_PX = 50
"""Cell size for the spatial reprojection-error heatmaps."""

RESIDUAL_COLORBAR_MAX_PX = 2.0
"""Upper bound on |residual| for all residual heatmap colorbars."""


def _orientation_label(yaw: float | None, pitch: float | None) -> str:
    """Short label like 'Y+0', 'P+30' for the stage heatmap axes."""
    if yaw is not None:
        return f"Y{yaw:+.0f}"
    assert pitch is not None
    return f"P{pitch:+.0f}"


def _solve_frame_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run solvePnP and return (rvec, tvec), or None if it fails."""
    success, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64),
        image_points.astype(np.float64).reshape(-1, 1, 2),
        camera_matrix.astype(np.float64),
        distortion.astype(np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return rvec, tvec


def _pose_to_yaw_pitch_distance(
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
) -> tuple[float, float, float]:
    """Convert a solvePnP result into (yaw_deg, pitch_deg, distance_m)."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    board_normal = rotation_matrix[:, 2]
    yaw_deg = float(np.degrees(np.arctan2(board_normal[0], -board_normal[2])))
    pitch_deg = float(np.degrees(np.arctan2(-board_normal[1], -board_normal[2])))
    center_in_board = object_points.astype(np.float64).mean(axis=0)
    center_in_camera = rotation_matrix @ center_in_board + tvec[:, 0]
    distance_m = float(np.linalg.norm(center_in_camera))
    return yaw_deg, pitch_deg, distance_m


def compute_per_frame_residuals(
    object_points_by_frame: list[np.ndarray],
    image_points_by_frame: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float, float] | None]]:
    """For each frame return (residuals_Nx2, observed_Nx2, pose)."""
    residuals_by_frame: list[np.ndarray] = []
    observed_by_frame: list[np.ndarray] = []
    poses_by_frame: list[tuple[float, float, float] | None] = []

    for object_points, image_points in zip(
        object_points_by_frame, image_points_by_frame, strict=False
    ):
        observed = image_points.reshape(-1, 2).astype(np.float64)
        pose = _solve_frame_pose(object_points, image_points, camera_matrix, distortion)
        if pose is None:
            residuals_by_frame.append(np.zeros_like(observed))
            observed_by_frame.append(observed)
            poses_by_frame.append(None)
            continue
        rvec, tvec = pose
        projected, _ = cv2.projectPoints(
            object_points.astype(np.float64),
            rvec,
            tvec,
            camera_matrix.astype(np.float64),
            distortion.astype(np.float64),
        )
        projected = projected.reshape(-1, 2)
        residuals_by_frame.append(observed - projected)
        observed_by_frame.append(observed)
        poses_by_frame.append(_pose_to_yaw_pitch_distance(rvec, tvec, object_points))

    return residuals_by_frame, observed_by_frame, poses_by_frame


def build_spatial_heatmap(
    residuals_by_frame: list[np.ndarray],
    observed_by_frame: list[np.ndarray],
    image_size: tuple[int, int],
    cell_size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_residual_mag, count) over a cell_size_px spatial grid."""
    width, height = image_size
    n_rows = height // cell_size_px
    n_cols = width // cell_size_px
    sum_mag = np.zeros((n_rows, n_cols), dtype=np.float64)
    count = np.zeros((n_rows, n_cols), dtype=np.int32)

    for residuals, observed in zip(residuals_by_frame, observed_by_frame, strict=False):
        magnitudes = np.linalg.norm(residuals, axis=1)
        xs = (observed[:, 0].astype(int)) // cell_size_px
        ys = (observed[:, 1].astype(int)) // cell_size_px
        valid = (xs >= 0) & (xs < n_cols) & (ys >= 0) & (ys < n_rows)
        np.add.at(sum_mag, (ys[valid], xs[valid]), magnitudes[valid])
        np.add.at(count, (ys[valid], xs[valid]), 1)

    mean = np.divide(sum_mag, count, out=np.full_like(sum_mag, np.nan), where=count > 0)
    return mean, count


def build_stage_heatmap(
    residuals_by_frame: list[np.ndarray],
    poses_by_frame: list[tuple[float, float, float] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_residual_mag, count) over the stage grid (distance × orientation)."""
    n_orientations = len(STAGE_ORIENTATIONS)
    n_distances = len(STAGE_DISTANCE_RANGES_M)
    sum_mag = np.zeros((n_distances, n_orientations), dtype=np.float64)
    count = np.zeros((n_distances, n_orientations), dtype=np.int32)

    for residuals, pose in zip(residuals_by_frame, poses_by_frame, strict=False):
        if pose is None:
            continue
        yaw, pitch, distance = pose
        magnitudes = np.linalg.norm(residuals, axis=1)
        total = magnitudes.sum()
        n = len(magnitudes)
        for stage_idx in range(len(STAGES)):
            if is_frame_in_stage(yaw, pitch, distance, stage_idx):
                row = stage_idx // n_orientations
                col = stage_idx % n_orientations
                sum_mag[row, col] += total
                count[row, col] += n

    mean = np.divide(sum_mag, count, out=np.full_like(sum_mag, np.nan), where=count > 0)
    return mean, count


def build_per_stage_spatial_heatmaps(
    residuals_by_frame: list[np.ndarray],
    observed_by_frame: list[np.ndarray],
    poses_by_frame: list[tuple[float, float, float] | None],
    image_size: tuple[int, int],
    cell_size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, count), each shape (n_stages, n_rows, n_cols)."""
    width, height = image_size
    n_rows = height // cell_size_px
    n_cols = width // cell_size_px
    n_stages = len(STAGES)
    sum_mag = np.zeros((n_stages, n_rows, n_cols), dtype=np.float64)
    count = np.zeros((n_stages, n_rows, n_cols), dtype=np.int32)

    for residuals, observed, pose in zip(
        residuals_by_frame, observed_by_frame, poses_by_frame, strict=False
    ):
        if pose is None:
            continue
        yaw, pitch, distance = pose
        magnitudes = np.linalg.norm(residuals, axis=1)
        xs = (observed[:, 0].astype(int)) // cell_size_px
        ys = (observed[:, 1].astype(int)) // cell_size_px
        valid = (xs >= 0) & (xs < n_cols) & (ys >= 0) & (ys < n_rows)
        if not valid.any():
            continue
        xs_v = xs[valid]
        ys_v = ys[valid]
        mags_v = magnitudes[valid]
        for stage_idx in range(n_stages):
            if is_frame_in_stage(yaw, pitch, distance, stage_idx):
                np.add.at(sum_mag[stage_idx], (ys_v, xs_v), mags_v)
                np.add.at(count[stage_idx], (ys_v, xs_v), 1)

    mean = np.divide(sum_mag, count, out=np.full_like(sum_mag, np.nan), where=count > 0)
    return mean, count


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _plot_spatial_axes(
    ax: plt.Axes,
    data: np.ndarray,
    cell_size_px: int,
    cmap: str,
    vmax: float,
    colorbar_label: str,
) -> None:
    extent = (0, data.shape[1] * cell_size_px, data.shape[0] * cell_size_px, 0)
    im = ax.imshow(
        data, extent=extent, cmap=cmap, vmin=0.0, vmax=vmax if vmax > 0 else 1.0, aspect="equal"
    )
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.85)


def _plot_stage_axes(
    ax: plt.Axes,
    values: np.ndarray,
    count: np.ndarray,
    cmap: str,
    vmax: float,
    colorbar_label: str,
    annotate_count: bool,
) -> None:
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    ax.set_xticks(range(len(STAGE_ORIENTATIONS)))
    ax.set_xticklabels([_orientation_label(y, p) for y, p in STAGE_ORIENTATIONS])
    ax.set_yticks(range(len(STAGE_DISTANCE_RANGES_M)))
    ax.set_yticklabels([f"{lo:.1f}–{hi:.1f}m" for lo, hi in STAGE_DISTANCE_RANGES_M])
    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.85)

    threshold = vmax * 0.6 if vmax > 0 else 0.5
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if count[i, j] == 0:
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="white")
                continue
            value = values[i, j]
            color = "white" if value < threshold else "black"
            if annotate_count:
                text = f"{value:.2f}\nn={count[i, j]}"
            else:
                text = f"n={count[i, j]}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)


def _plot_tiled_stage_spatial(
    fig: plt.Figure,
    outer_cell: SubplotSpec,
    per_stage_data: np.ndarray,
    cell_size_px: int,
    cmap: str,
    vmax: float,
    title: str,
    colorbar_label: str,
) -> None:
    """Fill outer_cell with a (title row) + (3×6 grid) of mini spatial heatmaps."""
    block = outer_cell.subgridspec(2, 1, height_ratios=[0.06, 1.0], hspace=0.05)

    title_ax = fig.add_subplot(block[0])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.2,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=title_ax.transAxes,
    )

    n_dist = len(STAGE_DISTANCE_RANGES_M)
    n_ori = len(STAGE_ORIENTATIONS)
    grid = block[1].subgridspec(
        n_dist, n_ori + 1, width_ratios=[1] * n_ori + [0.07], hspace=0.25, wspace=0.12
    )

    last_im = None
    for i in range(n_dist):
        for j in range(n_ori):
            ax = fig.add_subplot(grid[i, j])
            stage_idx = i * n_ori + j
            data = per_stage_data[stage_idx]
            extent = (0, data.shape[1] * cell_size_px, data.shape[0] * cell_size_px, 0)
            last_im = ax.imshow(
                data,
                extent=extent,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax if vmax > 0 else 1.0,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                yaw, pitch = STAGE_ORIENTATIONS[j]
                ax.set_title(_orientation_label(yaw, pitch), fontsize=10)
            if j == 0:
                lo, hi = STAGE_DISTANCE_RANGES_M[i]
                ax.set_ylabel(f"{lo:.1f}–{hi:.1f}m", fontsize=9)

    if last_im is not None:
        cbar_ax = fig.add_subplot(grid[:, n_ori])
        fig.colorbar(last_im, cax=cbar_ax, label=colorbar_label)


def plot_metrics(
    spatial_mean: np.ndarray,
    spatial_count: np.ndarray,
    stage_mean: np.ndarray,
    stage_count: np.ndarray,
    per_stage_mean: np.ndarray,
    per_stage_count: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    total_frames: int,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 22))
    outer = GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.22, height_ratios=[1.0, 1.0, 3.2])

    width, height = image_size

    spatial_count_vmax = float(spatial_count.max()) if spatial_count.size else 0.0
    stage_count_vmax = float(stage_count.max()) if stage_count.size else 0.0
    tiled_count_vmax = float(per_stage_count.max()) if per_stage_count.size else 0.0
    # All residual heatmaps share a fixed [0, RESIDUAL_COLORBAR_MAX_PX] scale so
    # they're directly comparable.
    residual_vmax = RESIDUAL_COLORBAR_MAX_PX

    # ── Row 0: overall spatial ──
    ax = fig.add_subplot(outer[0, 0])
    _plot_spatial_axes(
        ax, spatial_mean, SPATIAL_CELL_PX, "viridis", residual_vmax, "mean |residual| (px)"
    )
    ax.set_title(
        f"Spatial mean |residual|  (image {width}×{height}, "
        f"{SPATIAL_CELL_PX}px cells, RMS={rms:.3f}px)"
    )

    ax = fig.add_subplot(outer[0, 1])
    _plot_spatial_axes(
        ax,
        spatial_count.astype(float),
        SPATIAL_CELL_PX,
        "cividis",
        spatial_count_vmax,
        "observations",
    )
    ax.set_title(f"Spatial observation count  ({total_frames} frames)")

    # ── Row 1: overall stage ──
    ax = fig.add_subplot(outer[1, 0])
    _plot_stage_axes(
        ax,
        stage_mean,
        stage_count,
        "viridis",
        residual_vmax,
        "mean |residual| (px)",
        annotate_count=True,
    )
    ax.set_title("Stage mean |residual|")

    ax = fig.add_subplot(outer[1, 1])
    _plot_stage_axes(
        ax,
        stage_count.astype(float),
        stage_count,
        "cividis",
        stage_count_vmax,
        "observations",
        annotate_count=False,
    )
    ax.set_title("Stage observation count")

    # ── Row 2: tiled per-stage spatial ──
    _plot_tiled_stage_spatial(
        fig,
        outer[2, 0],
        per_stage_mean,
        SPATIAL_CELL_PX,
        cmap="viridis",
        vmax=residual_vmax,
        title="Per-stage spatial mean |residual|",
        colorbar_label="mean |residual| (px)",
    )
    _plot_tiled_stage_spatial(
        fig,
        outer[2, 1],
        per_stage_count.astype(float),
        SPATIAL_CELL_PX,
        cmap="cividis",
        vmax=tiled_count_vmax,
        title="Per-stage spatial observation count",
        colorbar_label="observations",
    )

    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections",
        type=Path,
        default=CALIBRATION_DETECTIONS_PATH,
        help="Path to calibration_detections.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: detections path with .png suffix).",
    )
    parser.add_argument(
        "--overwrite-yaml",
        action="store_true",
        help=f"Overwrite {CALIBRATION_YAML_PATH} with the stored calibration.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.detections.with_suffix(".png")

    data = np.load(args.detections, allow_pickle=True)
    object_points_by_frame = list(data["object_points"])
    image_points_by_frame = list(data["image_points"])
    image_size = (int(data["image_width"]), int(data["image_height"]))
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(data["distortion"], dtype=np.float64)
    rms = float(data["rms_error"])

    print(f"Loaded {len(object_points_by_frame)} frames from {args.detections}")
    print(f"Image size: {image_size[0]}×{image_size[1]}, RMS: {rms:.3f}px")

    residuals_by_frame, observed_by_frame, poses_by_frame = compute_per_frame_residuals(
        object_points_by_frame,
        image_points_by_frame,
        camera_matrix,
        distortion,
    )
    spatial_mean, spatial_count = build_spatial_heatmap(
        residuals_by_frame,
        observed_by_frame,
        image_size,
        SPATIAL_CELL_PX,
    )
    stage_mean, stage_count = build_stage_heatmap(residuals_by_frame, poses_by_frame)
    per_stage_mean, per_stage_count = build_per_stage_spatial_heatmaps(
        residuals_by_frame,
        observed_by_frame,
        poses_by_frame,
        image_size,
        SPATIAL_CELL_PX,
    )

    plot_metrics(
        spatial_mean,
        spatial_count,
        stage_mean,
        stage_count,
        per_stage_mean,
        per_stage_count,
        image_size,
        rms,
        len(object_points_by_frame),
        args.output,
    )
    print(f"Saved metrics PNG to {args.output}")

    if args.overwrite_yaml:
        result = CalibrationResult(
            camera_matrix=CameraMatrix.from_matrix(camera_matrix),
            distortion=DistortionCoefficients.from_array(distortion),
            image_size=ImageSize(width=image_size[0], height=image_size[1]),
            rms_error_pixels=rms,
        )
        result.save()
        print(f"Overwrote {CALIBRATION_YAML_PATH}")


if __name__ == "__main__":
    main()
