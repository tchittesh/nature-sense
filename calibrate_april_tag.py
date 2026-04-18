from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

from calibration_types import (
    CalibrationResult,
    CameraMatrix,
    DistortionCoefficients,
    ImageSize,
)
from constants import CALIBRATION_YAML_PATH, CAMERA_INDEX

CALIBRATION_DETECTIONS_PATH = CALIBRATION_YAML_PATH.with_name("calibration_detections.npz")
"""Where to dump raw AprilTag detections + calibration result for future metrics."""

# ── Board geometry (from camera_calibration_board.py) ─────────────────────────

TAG_WIDTH_M = 0.060
ADJACENT_TAG_CORNER_DISTANCE_M = 0.025
ADJACENT_TAG_CENTER_DISTANCE_M = TAG_WIDTH_M + ADJACENT_TAG_CORNER_DISTANCE_M

# ── Detection constants (from april_tag_corners.py) ───────────────────────────

CORNER_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
REFINE_CORNERS_WINDOW_RATIO = 1 / 15

N_TAGS = 36
EDGE_REJECT_PX = 5
MIN_IMAGES = 15
MIN_TAGS_PER_FRAME = 4

# ── Stage system ──────────────────────────────────────────────────────────────

# Orientations laid out left-to-right in the grid display. Exactly one of
# (yaw_target, pitch_target) is a float; the other is None and means that axis
# is unconstrained for this stage. This lets us split the old (0, 0) bucket
# into a "Y+0" axis constraint (yaw=0, pitch free) and a "P+0" axis constraint
# (pitch=0, yaw free).
STAGE_ORIENTATIONS: list[tuple[float | None, float | None]] = [
    (-30.0, None),  # Y-30
    (None, -30.0),  # P-30
    (0.0, None),  # Y+0
    (None, 0.0),  # P+0
    (None, 30.0),  # P+30
    (30.0, None),  # Y+30
]
STAGE_DISTANCE_RANGES_M: list[tuple[float, float]] = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]

STAGES: list[tuple[float | None, float | None, tuple[float, float]]] = [
    (yaw, pitch, distance_range)
    for distance_range in STAGE_DISTANCE_RANGES_M
    for yaw, pitch in STAGE_ORIENTATIONS
]

YAW_TOLERANCE_DEG = 15.0
PITCH_TOLERANCE_DEG = 15.0

CELL_SIZE_PX = 100
MIN_OBSERVATIONS_PER_CELL = 1
STAGE_COMPLETE_COVERAGE_RATIO = 0.90
AUTO_SAVE_MIN_NEW_CELLS = 3

# ── Detectors (from camera_calibration_board.py) ──────────────────────────────

DETECTORS = [
    Detector(
        families="tag36h11",
        nthreads=8,
        quad_decimate=quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    for quad_decimate in (1, 2, 4)
]


def get_tag_corner_points() -> np.ndarray:
    """Get 3D corner points for all tags in the board frame.

    Corner order:
          3---------2
          |         |
          |  TAG=N  |
          |         |
      ^   0---------1
    y |   <--60mm--->
      |             <-25mm->
      +--> x
    """
    corner_offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * TAG_WIDTH_M - (TAG_WIDTH_M // 2)
    tag_positions = np.mgrid[:6, :6].T.reshape(-1, 1, 2) * ADJACENT_TAG_CENTER_DISTANCE_M
    return np.concatenate(
        [tag_positions + corner_offsets, np.zeros([N_TAGS, 4, 1])], axis=-1
    ).astype(np.float32)


TAG_CORNER_POINTS = get_tag_corner_points()


def refine_corner(
    corners: np.ndarray,
    corner_index: int,
    gray_image: np.ndarray,
    min_refine_size: int = 3,
) -> np.ndarray:
    """Refine a single tag corner to subpixel accuracy."""
    my_corner = corners[corner_index]
    neighbor1 = corners[corner_index - 1]
    neighbor2 = corners[(corner_index + 1) % len(corners)]
    diff1 = np.abs(my_corner - neighbor1).max()
    diff2 = np.abs(my_corner - neighbor2).max()
    tag_side_len = max(diff1, diff2)
    refine_size = max(round(tag_side_len * REFINE_CORNERS_WINDOW_RATIO), min_refine_size)

    return cv2.cornerSubPix(
        gray_image,
        np.array([my_corner], dtype=np.float32),
        winSize=(refine_size, refine_size),
        zeroZone=(-1, -1),
        criteria=CORNER_SUBPIX_CRITERIA,
    )[0]


def detect_tags(
    gray_image: np.ndarray,
    edge_reject_px: int = EDGE_REJECT_PX,
    refine_corners: bool = True,
) -> list[tuple[int, np.ndarray]]:
    """Detect AprilTags and return (tag_id, corners) pairs."""
    all_tags: dict[int, Any] = {}
    for detector in DETECTORS:
        current_tags = detector.detect(
            gray_image, estimate_tag_pose=False, camera_params=None, tag_size=None
        )
        for tag in current_tags:
            if tag.tag_id not in all_tags and tag.tag_id < N_TAGS:
                all_tags[tag.tag_id] = tag

        if len(all_tags) == N_TAGS:
            break

    tags = list(all_tags.values())
    if len(tags) == 0:
        return []

    corners = np.stack([tag.corners for tag in tags], axis=0)

    image_bounds = np.array(gray_image.shape[::-1])
    is_valid = np.all(corners > edge_reject_px, axis=(1, 2))
    is_valid &= np.all(corners < (image_bounds - edge_reject_px), axis=(1, 2))

    corners = corners[is_valid]
    tag_ids = [tag.tag_id for tag, valid in zip(tags, is_valid, strict=False) if valid]

    if len(corners) == 0:
        return []

    if refine_corners:
        for i in range(corners.shape[0]):
            for j in range(corners.shape[1]):
                corners[i, j] = refine_corner(corners[i], j, gray_image)

    return list(zip(tag_ids, corners, strict=False))


def estimate_board_pose_approximate(
    tag_ids_and_corners: list[tuple[int, np.ndarray]],
    image_width: int,
    image_height: int,
) -> tuple[float | None, float | None, float | None]:
    """Estimate board yaw (deg), pitch (deg), and distance (m) using approximate intrinsics.

    Uses focal length = max(width, height) as a rough estimate. Accurate enough to guide
    the operator to the target pose; not used for the actual calibration.
    """
    focal_length = float(max(image_width, image_height))
    camera_matrix_approx = np.array(
        [
            [focal_length, 0.0, image_width / 2.0],
            [0.0, focal_length, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    tag_ids = [tag_id for tag_id, _ in tag_ids_and_corners]
    corners_stack = np.stack([corners for _, corners in tag_ids_and_corners], axis=0)
    object_points = TAG_CORNER_POINTS[tag_ids].reshape(-1, 3).astype(np.float64)
    image_points = corners_stack.reshape(-1, 1, 2).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix_approx,
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Board normal in camera frame: when board faces camera squarely, this ≈ [0, 0, -1]
    board_normal = rotation_matrix[:, 2]
    yaw_deg = float(np.degrees(np.arctan2(board_normal[0], -board_normal[2])))
    pitch_deg = float(np.degrees(np.arctan2(-board_normal[1], -board_normal[2])))

    board_center_in_board = object_points.mean(axis=0)
    board_center_in_camera = rotation_matrix @ board_center_in_board + tvec[:, 0]
    distance_m = float(np.linalg.norm(board_center_in_camera))

    return yaw_deg, pitch_deg, distance_m


def is_frame_in_stage(
    yaw_deg: float,
    pitch_deg: float,
    distance_m: float,
    stage_index: int,
) -> bool:
    """Return True if the measured pose matches the given stage.

    Each stage constrains exactly one axis: if target_yaw is not None we check
    only yaw; if target_pitch is not None we check only pitch. The other axis
    is ignored. Distance must always fall within the bucket.
    """
    target_yaw, target_pitch, (distance_min, distance_max) = STAGES[stage_index]
    if not (distance_min <= distance_m <= distance_max):
        return False
    if target_yaw is not None:
        return abs(yaw_deg - target_yaw) <= YAW_TOLERANCE_DEG
    assert target_pitch is not None
    return abs(pitch_deg - target_pitch) <= PITCH_TOLERANCE_DEG


def _build_cell_counts(
    image_points_list: list[np.ndarray],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Build a grid of full CELL_SIZE_PX×CELL_SIZE_PX cells.

    Any strip along the right/bottom edge that's smaller than one full cell is
    excluded, so coverage ratios are not diluted by remainder bands.
    """
    height, width = image_shape
    n_rows = height // CELL_SIZE_PX
    n_cols = width // CELL_SIZE_PX
    clean_height = n_rows * CELL_SIZE_PX
    clean_width = n_cols * CELL_SIZE_PX
    cell_counts = np.zeros((n_rows, n_cols), dtype=np.int32)
    for frame_points in image_points_list:
        for corner in frame_points:
            x, y = int(corner[0][0]), int(corner[0][1])
            if 0 <= x < clean_width and 0 <= y < clean_height:
                cell_counts[y // CELL_SIZE_PX, x // CELL_SIZE_PX] += 1
    return cell_counts


def compute_stage_coverage(
    stage_image_points: list[np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    """Return (covered_cells, total_cells).

    covered = cells with ≥2 corner observations; total = all CELL_SIZE_PX×CELL_SIZE_PX
    cells in the grid.
    """
    cell_counts = _build_cell_counts(stage_image_points, image_shape)
    return int(np.sum(cell_counts >= MIN_OBSERVATIONS_PER_CELL)), int(cell_counts.size)


def compute_new_contribution_cells(
    current_stage_image_points: list[np.ndarray],
    new_image_points: np.ndarray,
    image_shape: tuple[int, int],
) -> int:
    """Count not-yet-covered cells that new_image_points places at least one corner into.

    A cell counts as a "new contribution" if (a) it currently has fewer than
    MIN_OBSERVATIONS_PER_CELL observations AND (b) the candidate frame drops at
    least one corner into it. Whether the cell actually reaches the covered
    threshold after this frame does not matter.
    """
    cell_counts = _build_cell_counts(current_stage_image_points, image_shape)
    already_covered = cell_counts >= MIN_OBSERVATIONS_PER_CELL

    n_rows, n_cols = cell_counts.shape
    clean_height = n_rows * CELL_SIZE_PX
    clean_width = n_cols * CELL_SIZE_PX

    newly_touched = np.zeros_like(cell_counts, dtype=bool)
    for corner in new_image_points:
        x, y = int(corner[0][0]), int(corner[0][1])
        if 0 <= x < clean_width and 0 <= y < clean_height:
            newly_touched[y // CELL_SIZE_PX, x // CELL_SIZE_PX] = True

    return int(np.sum(newly_touched & ~already_covered))


def apply_heatmap_overlay(
    camera_frame: np.ndarray,
    stage_image_points: list[np.ndarray],
    alpha: float = 0.30,
    mirror: bool = False,
) -> np.ndarray:
    """Blend a green/red cell-based coverage overlay onto the camera frame.

    Each CELL_SIZE_PX×CELL_SIZE_PX cell is tinted green if covered
    (≥ MIN_OBSERVATIONS_PER_CELL corner observations) or red otherwise. Cell
    counts are computed from the stored (unmirrored) corner data; if
    mirror=True, the overlay is flipped to match a mirrored camera frame
    before blending.
    """
    height, width = camera_frame.shape[:2]
    cell_counts = _build_cell_counts(stage_image_points, (height, width))
    covered_mask = cell_counts >= MIN_OBSERVATIONS_PER_CELL

    overlay = np.zeros_like(camera_frame)
    n_rows, n_cols = covered_mask.shape
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * CELL_SIZE_PX
            y0 = row * CELL_SIZE_PX
            x1 = min(x0 + CELL_SIZE_PX, width)
            y1 = min(y0 + CELL_SIZE_PX, height)
            if y0 >= y1 or x0 >= x1:
                continue
            overlay[y0:y1, x0:x1] = (0, 200, 0) if covered_mask[row, col] else (0, 0, 200)

    if mirror:
        overlay = cv2.flip(overlay, 1)

    return cv2.addWeighted(camera_frame, 1.0 - alpha, overlay, alpha, 0)


def is_stage_complete(
    stage_image_points: list[np.ndarray],
    image_shape: tuple[int, int],
) -> bool:
    """True when covered/total ≥ STAGE_COMPLETE_COVERAGE_RATIO for the stage."""
    covered, total = compute_stage_coverage(stage_image_points, image_shape)
    return total > 0 and covered / total >= STAGE_COMPLETE_COVERAGE_RATIO


def get_matching_stage_indices(
    yaw_deg: float | None,
    pitch_deg: float | None,
    distance_m: float | None,
) -> list[int]:
    """Return all stage indices whose pose tolerance contains the given pose."""
    if yaw_deg is None or pitch_deg is None or distance_m is None:
        return []
    return [
        stage_idx
        for stage_idx in range(len(STAGES))
        if is_frame_in_stage(yaw_deg, pitch_deg, distance_m, stage_idx)
    ]


def dedupe_calibration_frames(
    object_points_by_stage: dict[int, list[np.ndarray]],
    image_points_by_stage: dict[int, list[np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect all unique (object_points, image_points) pairs across stages."""
    seen_ids: set[int] = set()
    all_object_points: list[np.ndarray] = []
    all_image_points: list[np.ndarray] = []
    for stage_idx in range(len(STAGES)):
        for object_points, image_points in zip(
            object_points_by_stage[stage_idx], image_points_by_stage[stage_idx], strict=False
        ):
            if id(image_points) not in seen_ids:
                seen_ids.add(id(image_points))
                all_object_points.append(object_points)
                all_image_points.append(image_points)
    return all_object_points, all_image_points


def count_unique_frames(image_points_by_stage: dict[int, list[np.ndarray]]) -> int:
    """Count unique frames across all stages (by array identity)."""
    seen_ids: set[int] = set()
    for frames in image_points_by_stage.values():
        for frame_points in frames:
            seen_ids.add(id(frame_points))
    return len(seen_ids)


def save_detections(
    path: Path,
    object_points_by_frame: list[np.ndarray],
    image_points_by_frame: list[np.ndarray],
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rms_error: float,
) -> None:
    """Dump raw detections + calibration result to a compressed .npz archive.

    Per-frame point arrays are stored as object arrays because each frame has
    a different number of detected tag corners. Load with
    ``data = np.load(path, allow_pickle=True)`` and iterate
    ``zip(data["object_points"], data["image_points"])`` to reproject.
    """
    width, height = image_size
    np.savez_compressed(
        path,
        object_points=np.array(object_points_by_frame, dtype=object),
        image_points=np.array(image_points_by_frame, dtype=object),
        image_width=np.int32(width),
        image_height=np.int32(height),
        camera_matrix=camera_matrix,
        distortion=distortion,
        rms_error=np.float64(rms_error),
    )


def get_nearest_incomplete_stage_index(
    yaw_deg: float | None,
    pitch_deg: float | None,
    distance_m: float | None,
    image_points_by_stage: dict[int, list[np.ndarray]],
    image_shape: tuple[int, int],
) -> int | None:
    """Pick the incomplete stage whose target is closest to the current pose."""
    incomplete = [
        i
        for i in range(len(STAGES))
        if not is_stage_complete(image_points_by_stage.get(i, []), image_shape)
    ]
    if not incomplete:
        return None
    if yaw_deg is None or pitch_deg is None or distance_m is None:
        return incomplete[0]

    def score(stage_idx: int) -> float:
        target_yaw, target_pitch, (distance_min, distance_max) = STAGES[stage_idx]
        distance_center = (distance_min + distance_max) / 2.0
        yaw_diff = (yaw_deg - target_yaw) if target_yaw is not None else 0.0
        pitch_diff = (pitch_deg - target_pitch) if target_pitch is not None else 0.0
        return (
            (yaw_diff / 20.0) ** 2
            + (pitch_diff / 20.0) ** 2
            + ((distance_m - distance_center) / 1.0) ** 2
        )

    return min(incomplete, key=score)


def _orientation_label(yaw: float | None, pitch: float | None) -> str:
    """Short label like 'Y+0', 'P+0', 'Y-30', 'P+30' for the grid."""
    if yaw is not None:
        return f"Y{yaw:+.0f}"
    assert pitch is not None
    return f"P{pitch:+.0f}"


def draw_gauge_panel(
    current_yaw: float | None,
    current_pitch: float | None,
    current_distance: float | None,
    target_stage_index: int | None,
    matched_stage_index: int | None,
    covered_cells: int,
    total_cells: int,
    total_frame_count: int,
    all_stages_done: bool,
) -> np.ndarray:
    """Panel showing current (or nearest) stage's pose gauges and coverage progress."""
    PANEL_W = 640
    PANEL_H = 330
    GAUGE_LEFT = 115
    GAUGE_RIGHT = PANEL_W - 115
    GAUGE_BAR_H = 14

    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # ── Header + derive target values (None → axis unconstrained) ─────────────
    target_yaw: float | None = None
    target_pitch: float | None = None
    distance_center = 1.0
    distance_half_width = 0.5

    if all_stages_done:
        cv2.putText(
            panel,
            "All stages complete!  Press 'c' to calibrate.",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 60),
            2,
        )
    elif target_stage_index is None:
        cv2.putText(
            panel,
            "Waiting for pose...",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 150, 150),
            2,
        )
    else:
        target_yaw, target_pitch, (distance_min, distance_max) = STAGES[target_stage_index]
        distance_center = (distance_min + distance_max) / 2.0
        distance_half_width = (distance_max - distance_min) / 2.0
        if target_stage_index == matched_stage_index:
            header = f"Matched Stage #{target_stage_index + 1}"
            header_color = (0, 220, 60)
        else:
            header = f"Guide \u2192 Stage #{target_stage_index + 1}"
            header_color = (0, 200, 255)
        cv2.putText(panel, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, header_color, 2)
        yaw_str = f"{target_yaw:+.0f}\u00b0" if target_yaw is not None else "any"
        pitch_str = f"{target_pitch:+.0f}\u00b0" if target_pitch is not None else "any"
        target_text = (
            f"Target:  Yaw={yaw_str}   "
            f"Pitch={pitch_str}   "
            f"Dist={distance_min:.1f}-{distance_max:.1f} m"
        )
        cv2.putText(panel, target_text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # ── Coverage progress bar (covered / total, target STAGE_COMPLETE_COVERAGE_RATIO) ─
    coverage_ratio = covered_cells / total_cells if total_cells > 0 else 0.0
    bar_x, bar_y, bar_w, bar_h = 10, 66, PANEL_W - 20, 18
    # Bar fills proportionally toward the 95% goal
    progress = min(coverage_ratio / STAGE_COMPLETE_COVERAGE_RATIO, 1.0)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        fill_color = (0, 200, 60) if progress >= 1.0 else (0, 140, 220)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), 1)
    cov_label = (
        f"Coverage: {covered_cells}/{total_cells} cells "
        f"({coverage_ratio * 100:.0f}% \u2192 {STAGE_COMPLETE_COVERAGE_RATIO * 100:.0f}%)"
    )
    cv2.putText(
        panel,
        cov_label,
        (bar_x + 4, bar_y + bar_h - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (220, 220, 220),
        1,
    )

    # ── Gauge bars ────────────────────────────────────────────────────────────
    def draw_gauge(
        y_center: int,
        label: str,
        current: float | None,
        target: float | None,
        val_min: float,
        val_max: float,
        tolerance: float,
        unit: str,
    ) -> None:
        gauge_w = GAUGE_RIGHT - GAUGE_LEFT

        def to_px(v: float) -> int:
            return int(
                GAUGE_LEFT + np.clip((v - val_min) / (val_max - val_min), 0.0, 1.0) * gauge_w
            )

        top = y_center - GAUGE_BAR_H // 2
        bot = y_center + GAUGE_BAR_H // 2

        cv2.rectangle(panel, (GAUGE_LEFT, top), (GAUGE_RIGHT, bot), (40, 40, 40), -1)

        if target is not None:
            cv2.rectangle(
                panel,
                (to_px(target - tolerance), top),
                (to_px(target + tolerance), bot),
                (0, 80, 0),
                -1,
            )
            t_px = to_px(target)
            cv2.line(panel, (t_px, top - 4), (t_px, bot + 4), (0, 200, 0), 2)

        if current is not None:
            if target is not None:
                in_range = abs(current - target) <= tolerance
                dot_color = (0, 255, 80) if in_range else (30, 100, 255)
            else:
                dot_color = (200, 200, 200)
            cv2.circle(panel, (to_px(current), y_center), GAUGE_BAR_H // 2 + 3, dot_color, -1)
            val_str = f"{current:+.1f}{unit}" if unit == "\u00b0" else f"{current:.2f} {unit}"
            cv2.putText(
                panel,
                val_str,
                (GAUGE_RIGHT + 8, y_center + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                dot_color,
                1,
            )
        else:
            cv2.putText(
                panel,
                "---",
                (GAUGE_RIGHT + 8, y_center + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (90, 90, 90),
                1,
            )

        cv2.putText(
            panel, label, (6, y_center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        if target is not None:
            t_label = f"T:{target:+.0f}{unit}" if unit == "\u00b0" else f"T:{target:.1f}{unit}"
        else:
            t_label = "T: any"
        cv2.putText(
            panel, t_label, (6, y_center + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1
        )

    draw_gauge(138, "YAW", current_yaw, target_yaw, -65.0, 65.0, YAW_TOLERANCE_DEG, "\u00b0")
    draw_gauge(
        198, "PITCH", current_pitch, target_pitch, -65.0, 65.0, PITCH_TOLERANCE_DEG, "\u00b0"
    )
    draw_gauge(258, "DIST", current_distance, distance_center, 0.0, 4.5, distance_half_width, "m")

    # ── Footer ────────────────────────────────────────────────────────────────
    cv2.putText(
        panel,
        "c=calibrate  u=undistort  q=quit",
        (10, PANEL_H - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        panel,
        f"Total calibration frames: {total_frame_count}",
        (10, PANEL_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (130, 130, 130),
        1,
    )

    return panel


def draw_stage_grid(
    image_points_by_stage: dict[int, list[np.ndarray]],
    matched_stage_indices: list[int],
    target_stage_index: int | None,
    image_shape: tuple[int, int],
    total_frame_count: int,
    calibration: CalibrationResult | None,
) -> np.ndarray:
    """3×9 grid of all stages (rows = distance bucket, cols = orientation)."""
    HEADER_H = 30
    ROW_HEADER_W = 92
    CELL_W = 94
    CELL_H = 62
    FOOTER_H = 26
    PANEL_W = ROW_HEADER_W + CELL_W * len(STAGE_ORIENTATIONS)
    PANEL_H = HEADER_H + CELL_H * len(STAGE_DISTANCE_RANGES_M) + FOOTER_H

    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # Column headers (orientations)
    for col, (yaw, pitch) in enumerate(STAGE_ORIENTATIONS):
        label = _orientation_label(yaw, pitch)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = ROW_HEADER_W + col * CELL_W + (CELL_W - text_size[0]) // 2
        cv2.putText(panel, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Row headers (distance buckets)
    for row, (distance_min, distance_max) in enumerate(STAGE_DISTANCE_RANGES_M):
        label = f"{distance_min:.1f}-{distance_max:.1f}m"
        y = HEADER_H + row * CELL_H + CELL_H // 2 + 4
        cv2.putText(panel, label, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Cells
    n_orientations = len(STAGE_ORIENTATIONS)
    for stage_idx in range(len(STAGES)):
        row = stage_idx // n_orientations
        col = stage_idx % n_orientations
        x0 = ROW_HEADER_W + col * CELL_W
        y0 = HEADER_H + row * CELL_H
        x1 = x0 + CELL_W
        y1 = y0 + CELL_H

        stage_points = image_points_by_stage.get(stage_idx, [])
        covered, total = compute_stage_coverage(stage_points, image_shape)
        ratio = covered / total if total > 0 else 0.0
        complete = is_stage_complete(stage_points, image_shape)

        if complete:
            bg_color = (30, 130, 30)
        elif covered == 0:
            bg_color = (35, 35, 35)
        elif ratio >= STAGE_COMPLETE_COVERAGE_RATIO / 2:
            bg_color = (30, 130, 200)
        else:
            bg_color = (40, 50, 140)
        cv2.rectangle(panel, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), bg_color, -1)

        if stage_idx in matched_stage_indices:
            border_color = (255, 255, 255)
            border_thickness = 2
        elif stage_idx == target_stage_index:
            border_color = (0, 220, 220)
            border_thickness = 2
        else:
            border_color = (80, 80, 80)
            border_thickness = 1
        cv2.rectangle(panel, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), border_color, border_thickness)

        cv2.putText(
            panel,
            f"#{stage_idx + 1}",
            (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (230, 230, 230),
            1,
        )

        pct_text = f"{ratio * 100:.0f}%" if total > 0 else "-"
        text_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = x0 + (CELL_W - text_size[0]) // 2
        ty = y0 + CELL_H // 2 + 6
        cv2.putText(panel, pct_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if total > 0:
            sub_text = f"{covered}/{total}"
            sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            sx = x0 + (CELL_W - sub_size[0]) // 2
            cv2.putText(
                panel, sub_text, (sx, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1
            )

    # Footer
    n_complete = sum(
        1
        for i in range(len(STAGES))
        if is_stage_complete(image_points_by_stage.get(i, []), image_shape)
    )
    footer_text = f"Stages complete: {n_complete}/{len(STAGES)}   |   Frames: {total_frame_count}"
    if calibration is not None:
        footer_text += f"   |   RMS: {calibration.rms_error_pixels:.3f}px"
    cv2.putText(
        panel, footer_text, (10, PANEL_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
    )

    return panel


def main() -> None:
    """Opportunistic camera calibration using an AprilTag board across 27 pose buckets."""
    object_points_by_stage: dict[int, list[np.ndarray]] = {i: [] for i in range(len(STAGES))}
    image_points_by_stage: dict[int, list[np.ndarray]] = {i: [] for i in range(len(STAGES))}

    cap = cv2.VideoCapture(CAMERA_INDEX)
    calibration: CalibrationResult | None = None
    show_undistorted = False
    prev_all_stages_done = False

    print("Camera Calibration - AprilTag Board (Opportunistic)")
    print("----------------------------------------------------")
    print(f"Board: 6×6 tag36h11 grid ({N_TAGS} tags)")
    print(
        f"Tag size: {TAG_WIDTH_M * 1000:.0f}mm, gap: {ADJACENT_TAG_CORNER_DISTANCE_M * 1000:.0f}mm"
    )
    print(f"Stages: {len(STAGES)}  |  Min tags/frame: {MIN_TAGS_PER_FRAME}")
    print(
        "\nMove the board around; frames auto-save when they contribute ≥"
        f"{AUTO_SAVE_MIN_NEW_CELLS} new covered cells to a matched stage."
    )
    print("Controls: c=calibrate  u=undistort  q=quit  (Ctrl+C to abort safely)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_height, image_width = gray.shape

            tag_ids_and_corners = detect_tags(gray)
            n_detected = len(tag_ids_and_corners)

            yaw: float | None = None
            pitch: float | None = None
            distance: float | None = None
            if n_detected >= MIN_TAGS_PER_FRAME:
                yaw, pitch, distance = estimate_board_pose_approximate(
                    tag_ids_and_corners, image_width, image_height
                )

            matched_stage_indices = get_matching_stage_indices(yaw, pitch, distance)

            # ── Auto-save: each matched stage independently accepts the frame if it
            # gains ≥AUTO_SAVE_MIN_NEW_CELLS new covered cells. The same ndarray is
            # shared across stages, so dedupe_calibration_frames() collapses it to
            # one entry at calibration time.
            if matched_stage_indices:
                tag_ids = [tag_id for tag_id, _ in tag_ids_and_corners]
                corners_stack = np.stack([corners for _, corners in tag_ids_and_corners], axis=0)
                candidate_image_points = corners_stack.reshape(-1, 1, 2).astype(np.float32)
                candidate_object_points = TAG_CORNER_POINTS[tag_ids].reshape(-1, 3)

                saved_to_stages: list[tuple[int, int]] = []
                for stage_idx in matched_stage_indices:
                    new_cells = compute_new_contribution_cells(
                        image_points_by_stage[stage_idx], candidate_image_points, gray.shape
                    )
                    if new_cells >= AUTO_SAVE_MIN_NEW_CELLS:
                        image_points_by_stage[stage_idx].append(candidate_image_points)
                        object_points_by_stage[stage_idx].append(candidate_object_points)
                        saved_to_stages.append((stage_idx, new_cells))

                if saved_to_stages:
                    stage_summary = ", ".join(f"#{s + 1}(+{n})" for s, n in saved_to_stages)
                    print(
                        f"Auto-saved frame {count_unique_frames(image_points_by_stage)} "
                        f"({n_detected} tags) \u2192 stage(s) {stage_summary}"
                    )

            # ── Global stage-completion state ─────────────────────────────────────
            all_stages_done = all(
                is_stage_complete(image_points_by_stage[i], gray.shape) for i in range(len(STAGES))
            )
            if all_stages_done and not prev_all_stages_done:
                print(f"\nAll {len(STAGES)} stages complete! Press 'c' to compute calibration.")
            prev_all_stages_done = all_stages_done

            # ── Pick primary matched stage: prefer incomplete, else first match ───
            primary_matched_stage: int | None = None
            if matched_stage_indices:
                incomplete_matched = [
                    s
                    for s in matched_stage_indices
                    if not is_stage_complete(image_points_by_stage[s], gray.shape)
                ]
                primary_matched_stage = (
                    incomplete_matched[0] if incomplete_matched else matched_stage_indices[0]
                )

            # ── Determine target stage: primary matched, else nearest incomplete ──
            target_stage_index: int | None
            if primary_matched_stage is not None:
                target_stage_index = primary_matched_stage
            elif not all_stages_done:
                target_stage_index = get_nearest_incomplete_stage_index(
                    yaw, pitch, distance, image_points_by_stage, gray.shape
                )
            else:
                target_stage_index = None

            total_frame_count = count_unique_frames(image_points_by_stage)

            # ── Camera feed: undistort (optional) → mirror → cell heatmap → tags → text
            display_frame = frame.copy()
            if show_undistorted and calibration is not None:
                camera_matrix_cv, dist_cv = calibration.get_opencv_params()
                display_frame = cv2.undistort(display_frame, camera_matrix_cv, dist_cv)
            display_frame = cv2.flip(display_frame, 1)

            overlay_stage: int | None = (
                primary_matched_stage if primary_matched_stage is not None else target_stage_index
            )
            if overlay_stage is not None:
                display_frame = apply_heatmap_overlay(
                    display_frame, image_points_by_stage[overlay_stage], mirror=True
                )

            if n_detected > 0:
                outline_color = (0, 255, 0) if matched_stage_indices else (0, 140, 255)
                for tag_id, tag_corners in tag_ids_and_corners:
                    mirrored_corners = tag_corners.copy()
                    mirrored_corners[:, 0] = image_width - 1 - mirrored_corners[:, 0]
                    pts = mirrored_corners.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, outline_color, 1)
                    for corner in mirrored_corners:
                        cv2.circle(
                            display_frame, (int(corner[0]), int(corner[1])), 3, outline_color, -1
                        )
                    center = mirrored_corners.mean(axis=0).astype(int)
                    cv2.putText(
                        display_frame,
                        str(tag_id),
                        tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

            if all_stages_done:
                status = "All stages complete - press 'c' to calibrate"
            elif matched_stage_indices:
                stage_list = ", ".join(f"#{s + 1}" for s in matched_stage_indices)
                status = f"Matched stage(s) {stage_list} - move for coverage"
            elif n_detected < MIN_TAGS_PER_FRAME:
                status = f"Detected {n_detected}/{MIN_TAGS_PER_FRAME} tags - need more"
            elif target_stage_index is not None:
                target_yaw, target_pitch, (distance_min, distance_max) = STAGES[target_stage_index]
                yaw_str = f"{target_yaw:+.0f}\u00b0" if target_yaw is not None else "any"
                pitch_str = f"{target_pitch:+.0f}\u00b0" if target_pitch is not None else "any"
                status = (
                    f"Move \u2192 stage #{target_stage_index + 1}  "
                    f"Y={yaw_str} P={pitch_str} "
                    f"D={distance_min:.1f}-{distance_max:.1f}m"
                )
            else:
                status = "No target"

            cv2.putText(
                display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
            )
            if calibration is not None:
                cv2.putText(
                    display_frame,
                    f"RMS: {calibration.rms_error_pixels:.3f}px",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Camera Calibration - AprilTag", display_frame)

            # ── Gauge panel (target stage) ────────────────────────────────────────
            covered_cells = 0
            total_cells = 0
            if target_stage_index is not None:
                covered_cells, total_cells = compute_stage_coverage(
                    image_points_by_stage[target_stage_index], gray.shape
                )
            cv2.imshow(
                "Stage Gauge",
                draw_gauge_panel(
                    yaw,
                    pitch,
                    distance,
                    target_stage_index,
                    primary_matched_stage,
                    covered_cells,
                    total_cells,
                    total_frame_count,
                    all_stages_done,
                ),
            )

            # ── Stage grid (all 27 stages at a glance) ────────────────────────────
            cv2.imshow(
                "Stage Grid",
                draw_stage_grid(
                    image_points_by_stage,
                    matched_stage_indices,
                    target_stage_index,
                    gray.shape,
                    total_frame_count,
                    calibration,
                ),
            )

            # ── Key handling ──────────────────────────────────────────────────────
            key = cv2.waitKey(1000) & 0xFF

            if key == ord("c") and total_frame_count >= MIN_IMAGES:
                print("Computing calibration...")
                height, width = gray.shape
                all_object_points, all_image_points = dedupe_calibration_frames(
                    object_points_by_stage, image_points_by_stage
                )
                print(f"Using {len(all_image_points)} unique frames.")
                rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                    all_object_points, all_image_points, (width, height), None, None
                )
                calibration = CalibrationResult(
                    camera_matrix=CameraMatrix.from_matrix(camera_matrix),
                    distortion=DistortionCoefficients.from_array(dist_coeffs),
                    image_size=ImageSize(width=width, height=height),
                    rms_error_pixels=rms,
                )
                print(f"Calibration complete! RMS error: {rms:.3f} pixels")
                calibration.save()
                print(f"Saved calibration to {CALIBRATION_YAML_PATH}")
                save_detections(
                    CALIBRATION_DETECTIONS_PATH,
                    all_object_points,
                    all_image_points,
                    (width, height),
                    camera_matrix,
                    dist_coeffs,
                    rms,
                )
                print(f"Saved detections to {CALIBRATION_DETECTIONS_PATH}")

            elif key == ord("u") and calibration is not None:
                show_undistorted = not show_undistorted

            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
