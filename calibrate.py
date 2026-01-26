from __future__ import annotations

import argparse
import cv2
import numpy as np

from calibration_types import (
    CameraMatrix,
    DistortionCoefficients,
    ImageSize,
    CalibrationResult,
)
from constants import CALIBRATION_YAML_PATH, CAMERA_INDEX

CHECKERBOARD_SIZE = (9, 6)  # Internal corners (cols, rows)
SQUARE_SIZE_MM = 20.0
MIN_IMAGES = 15

def create_coverage_visualization(img_points: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    """Create a visualization showing coverage of calibration points across the image.

    Args:
        img_points: List of corner point arrays from saved calibration images
        image_shape: (height, width) of the camera image

    Returns:
        BGR image showing coverage heatmap and individual point locations
    """
    height, width = image_shape

    # Create a blank canvas
    coverage = np.zeros((height, width, 3), dtype=np.uint8)

    if len(img_points) == 0:
        # No points yet - show instructions
        cv2.putText(coverage, "No calibration images yet",
                    (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100, 100, 100), 2)
        cv2.putText(coverage, "Capture images with 's' key",
                    (width//4, height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100, 100, 100), 2)
        return coverage

    # Create a heatmap by accumulating all points
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Draw each set of corners with uniform intensity
    # Color will represent number of observations
    for corners in img_points:
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            if 0 <= x < width and 0 <= y < height:
                # Draw a circle for each point, accumulating intensity
                cv2.circle(heatmap, (x, y), 15, 1.0, -1)

    # Normalize and convert to color
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend with black background
    coverage = heatmap_colored

    # Draw grid lines to show regions
    grid_color = (50, 50, 50)
    for i in range(1, 3):
        cv2.line(coverage, (i * width // 3, 0), (i * width // 3, height), grid_color, 1)
        cv2.line(coverage, (0, i * height // 3), (width, i * height // 3), grid_color, 1)

    return coverage

def create_checkerboard_pattern(dpi: float) -> tuple[np.ndarray, int, float]:
    """Create a checkerboard pattern image at correct physical size.

    Args:
        dpi: Screen DPI to use for calculating pixel dimensions.

    Returns:
        Tuple of (checkerboard image in BGR, square size in pixels, actual square size in mm)
    """
    square_size_pixels = int(SQUARE_SIZE_MM * dpi / 25.4)  # 1 inch = 25.4 mm

    # Calculate actual square size after quantization
    actual_square_size_mm = square_size_pixels * 25.4 / dpi

    print(f"Square size: {SQUARE_SIZE_MM}mm (nominal) -> {square_size_pixels}px -> {actual_square_size_mm:.4f}mm (actual) at {dpi} DPI")

    # Number of squares (add 1 to internal corners for outer squares)
    num_squares_x = CHECKERBOARD_SIZE[0] + 1
    num_squares_y = CHECKERBOARD_SIZE[1] + 1

    # Create checkerboard pattern
    width_pixels = num_squares_x * square_size_pixels
    height_pixels = num_squares_y * square_size_pixels

    checkerboard = np.zeros((height_pixels, width_pixels), dtype=np.uint8)

    for i in range(num_squares_y):
        for j in range(num_squares_x):
            # Alternate black (0) and white (255)
            if (i + j) % 2 == 0:
                y_start = i * square_size_pixels
                y_end = (i + 1) * square_size_pixels
                x_start = j * square_size_pixels
                x_end = (j + 1) * square_size_pixels
                checkerboard[y_start:y_end, x_start:x_end] = 255

    # Convert to BGR for OpenCV
    checkerboard_bgr = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)

    # Add white border around the pattern
    border_size = int(square_size_pixels/2)  # Border width equal to one square
    checkerboard_with_border = cv2.copyMakeBorder(
        checkerboard_bgr,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)  # White border
    )

    return checkerboard_with_border, square_size_pixels, actual_square_size_mm

def display_calibration_pattern(dpi: float) -> float:
    """Display the calibration pattern in a separate window.

    Args:
        dpi: Screen DPI to use for pattern display.

    Returns:
        Actual square size in mm after pixel quantization.
    """
    pattern, square_size_pixels, actual_square_size_mm = create_checkerboard_pattern(dpi)

    window_name = "Calibration Pattern - Display on Screen"
    # Use WINDOW_AUTOSIZE to prevent resizing and maintain exact pixel dimensions
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.imshow(window_name, pattern)

    height, width = pattern.shape[:2]
    print(f"\nCalibration pattern displayed at {width}x{height}px")
    print(f"Pattern: {CHECKERBOARD_SIZE[0]+1}x{CHECKERBOARD_SIZE[1]+1} squares")
    print(f"Each square: {actual_square_size_mm:.4f}mm ({square_size_pixels}px)")
    print("Position this window on your screen or secondary display for calibration\n")

    return actual_square_size_mm

def main() -> None:
    """Interactive camera calibration."""
    parser = argparse.ArgumentParser(description="Interactive camera calibration tool")
    parser.add_argument(
        "--dpi",
        type=float,
        help="Screen DPI for on-screen pattern display. If provided, displays calibration checkerboard."
    )
    args = parser.parse_args()

    # Display calibration pattern if DPI provided and get actual square size
    if args.dpi is not None:
        actual_square_size_mm = display_calibration_pattern(args.dpi)
    else:
        actual_square_size_mm = SQUARE_SIZE_MM

    # Storage for calibration data
    obj_points: list[np.ndarray] = []  # 3D world points
    img_points: list[np.ndarray] = []  # 2D image points

    # Prepare object points (0,0,0), (1,0,0), (2,0,0)...
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= actual_square_size_mm

    cap = cv2.VideoCapture(CAMERA_INDEX)
    calibration: CalibrationResult | None = None
    show_undistorted = False

    print("Camera Calibration")
    print("------------------")
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners")
    print(f"Square size: {actual_square_size_mm:.4f}mm")

    print("\nControls:")
    print("  s - Save frame (when corners detected)")
    print("  c - Compute calibration (need >= 15 images)")
    print("  u - Toggle undistorted view")
    print("  q - Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

        if found:
            # Refine corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, found)
            status = "Corners detected - press 's' to save"
        else:
            status = "No corners detected"

        # Apply undistortion if calibrated and toggled
        if calibration and show_undistorted:
            mtx, dist = calibration.get_opencv_params()
            frame = cv2.undistort(frame, mtx, dist)
            status = "Undistorted view"

        # Draw status
        cv2.putText(frame, f"Images: {len(img_points)}/{MIN_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if calibration is not None:
            cv2.putText(frame, f"RMS Error: {calibration.rms_error_pixels:.3f}px", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera Calibration", frame)

        # Always show coverage visualization
        coverage_viz = create_coverage_visualization(img_points, gray.shape)
        cv2.imshow("Coverage Visualization", coverage_viz)

        WAIT_TIME_MS = 1000
        key = cv2.waitKey(WAIT_TIME_MS) & 0xFF

        if key == ord('s') and found:
            obj_points.append(objp)
            img_points.append(corners)
            print(f"Saved image {len(img_points)}")

        elif key == ord('c') and len(img_points) >= MIN_IMAGES:
            print("Computing calibration...")
            height, width = gray.shape
            rms, mtx, dist, _, _ = cv2.calibrateCamera(
                obj_points, img_points, (width, height), None, None)
            calibration = CalibrationResult(
                camera_matrix=CameraMatrix.from_matrix(mtx),
                distortion=DistortionCoefficients.from_array(dist),
                image_size=ImageSize(width=width, height=height),
                rms_error_pixels=rms
            )
            print(f"Calibration complete! RMS error: {rms:.3f} pixels")

        elif key == ord('u') and calibration:
            show_undistorted = not show_undistorted

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if calibration is not None:
        calibration.save()
        print(f"Saved calibration to {CALIBRATION_YAML_PATH}")

if __name__ == '__main__':
    main()
