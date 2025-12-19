import cv2
import numpy as np
import os
from glob import glob
from ultralytics import YOLO

# ------------- Lane detection (with smoothing) ------------- #

def color_filter_lane(image):
    """
    Keep mostly white and yellow lane colors using HLS.
    This reduces noise from shadows and road texture.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # White mask
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # Yellow mask
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered = cv2.bitwise_and(image, image, mask=combined_mask)
    return filtered


def canny_edge_detector(image):
    """Color filter + grayscale + blur + Canny edges."""
    filtered = color_filter_lane(image)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(edges):
    """
    Focus on the lower-middle of the frame where lanes usually appear.
    You can fine-tune these ratios per video.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(0.10 * width), height),
        (int(0.42 * width), int(0.62 * height)),
        (int(0.58 * width), int(0.62 * height)),
        (int(0.90 * width), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)
    return masked


def detect_lines(edges):
    """Probabilistic Hough Transform to get line segments."""
    lines = cv2.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=120
    )
    return lines


def average_slope_intercept(image, lines):
    """
    Turn many short line segments into one left and one right lane line.
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    height, width, _ = image.shape

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        intercept = y1 - slope * x1

        # Ignore almost flat lines
        if abs(slope) < 0.4:
            continue

        # Left lane: negative slope, on left half
        if slope < 0 and x1 < width / 2 and x2 < width / 2:
            left_fit.append((slope, intercept))
        # Right lane: positive slope, on right half
        elif slope > 0 and x1 > width / 2 and x2 > width / 2:
            right_fit.append((slope, intercept))

    lane_lines = []

    if left_fit:
        left_avg = np.mean(left_fit, axis=0)
        lane_lines.append(make_points(image, left_avg))
    if right_fit:
        right_avg = np.mean(right_fit, axis=0)
        lane_lines.append(make_points(image, right_avg))

    if not lane_lines:
        return None

    return lane_lines


def make_points(image, line_params):
    """Convert (slope, intercept) to endpoints to draw."""
    slope, intercept = line_params
    height, width, _ = image.shape

    y1 = height
    y2 = int(height * 0.62)

    x1 = int((y1 - intercept) / (slope + 1e-6))
    x2 = int((y2 - intercept) / (slope + 1e-6))

    return np.array([x1, y1, x2, y2])


def smooth_lane_lines(image, lane_lines, prev_left, prev_right, alpha=0.9):
    """
    Smooth lanes over time using exponential moving average.
    alpha close to 1 = more weight on previous frames (less shaking).
    """
    height, width, _ = image.shape

    new_left = None
    new_right = None

    # Separate current left/right lines (if we have any)
    if lane_lines is not None:
        for line in lane_lines:
            x1, y1, x2, y2 = line
            # middle x of the line
            mid_x = (x1 + x2) / 2
            if mid_x < width / 2:
                new_left = line
            else:
                new_right = line

    def blend(prev, new):
        if prev is not None and new is not None:
            return (alpha * prev + (1 - alpha) * new).astype(int)
        elif new is not None:
            return new
        else:
            return prev

    sm_left = blend(prev_left, new_left)
    sm_right = blend(prev_right, new_right)

    smoothed_lines = []
    if sm_left is not None:
        smoothed_lines.append(sm_left)
    if sm_right is not None:
        smoothed_lines.append(sm_right)

    return smoothed_lines if smoothed_lines else None, sm_left, sm_right


def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=10):
    """Overlay lane lines on the frame."""
    if lines is None:
        return image

    line_img = np.zeros_like(image)
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    combined = cv2.addWeighted(image, 0.85, line_img, 1.0, 1)
    return combined

# ------------- YOLO object detection ------------- #

def load_yolo():
    """Load YOLOv8 nano model."""
    model = YOLO("yolov8n.pt")
    return model

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}
PERSON_CLASSES = {"person"}

def draw_detections(image, results, model_names, conf_threshold=0.4):
    """
    Draw bounding boxes for vehicles and people.
    """
    annotated = image.copy()
    if not results:
        return annotated

    boxes = results.boxes
    if boxes is None:
        return annotated

    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        conf_val = float(conf)
        if conf_val < conf_threshold:
            continue

        class_id = int(cls)
        class_name = model_names[class_id]

        if class_name not in VEHICLE_CLASSES and class_name not in PERSON_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box)

        if class_name in VEHICLE_CLASSES:
            color = (255, 0, 0)    # blue for vehicles
        else:
            color = (0, 255, 255)  # yellow for people

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf_val:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    return annotated

# ------------- Main processing ------------- #

def process_single_video(video_path, output_path, model, show=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model_names = model.names

    print(f"[INFO] Processing {os.path.basename(video_path)} ...")

    # Previous lane lines for smoothing
    prev_left = None
    prev_right = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Lane detection
        edges = canny_edge_detector(frame)
        roi = region_of_interest(edges)
        lines = detect_lines(roi)
        raw_lane_lines = average_slope_intercept(frame, lines)

        # Temporal smoothing
        smoothed_lanes, prev_left, prev_right = smooth_lane_lines(
            frame, raw_lane_lines, prev_left, prev_right, alpha=0.9
        )

        frame_with_lanes = draw_lane_lines(frame, smoothed_lanes)

        # 2) Object detection with YOLO
        yolo_result = model(frame_with_lanes, verbose=False)[0]
        frame_final = draw_detections(frame_with_lanes, yolo_result, model_names)

        out.write(frame_final)

        if show:
            cv2.imshow("Lane + Object Detection", frame_final)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"[INFO] Saved: {output_path}")


def process_all_videos(video_dir="videos", output_dir="outputs", show=True):
    os.makedirs(output_dir, exist_ok=True)
    model = load_yolo()

    video_files = glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"[WARNING] No .mp4 files found in {video_dir}")
        return

    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        process_single_video(video_path, output_path, model, show=show)


if __name__ == "__main__":
    process_all_videos(video_dir="videos", output_dir="outputs", show=True)
