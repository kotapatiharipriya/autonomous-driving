import cv2
import numpy as np

# ---------- Helper functions ----------

def canny_edge_detector(image):
    """Apply grayscale, blur and Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    """
    Apply a mask to keep only the region where lanes usually appear.
    This is a polygon covering the bottom half of the frame.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # You can tweak this polygon depending on your video.
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_lines(edges):
    """
    Use Hough Transform to detect line segments.
    Returns an array of line segments.
    """
    # rho=2, theta=1 deg, threshold=50, min line length, max line gap
    lines = cv2.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    return lines

def average_slope_intercept(image, lines):
    """
    Take all Hough lines and average them into one left and one right line.
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    height, width, _ = image.shape

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Avoid division by zero
        if x2 == 0 and x1 == 0:
            continue
        # Fit a first degree polynomial (line) y = mx + b
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        intercept = y1 - slope * x1

        # Filter out almost horizontal lines
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lane_lines = []
    if left_fit:
        left_fit_avg = np.average(left_fit, axis=0)
        lane_lines.append(make_points(image, left_fit_avg))
    if right_fit:
        right_fit_avg = np.average(right_fit, axis=0)
        lane_lines.append(make_points(image, right_fit_avg))

    return lane_lines

def make_points(image, line_params):
    """
    Convert slope & intercept to pixel coordinates for drawing.
    """
    slope, intercept = line_params
    height, width, _ = image.shape

    y1 = height           # bottom of the image
    y2 = int(height * 0.6) # somewhere above

    # x = (y - b) / m
    x1 = int((y1 - intercept) / (slope + 1e-6))
    x2 = int((y2 - intercept) / (slope + 1e-6))

    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines, color=(0, 255, 0), thickness=10):
    """
    Draw lane lines on a blank image, then overlay on the original.
    """
    if lines is None:
        return image

    line_image = np.zeros_like(image)

    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    # Overlay with some transparency
    combined = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined

# ---------- Main video processing ----------

def process_video(input_path, output_path=None, show=True):
    cap = cv2.VideoCapture(input_path)

    # Optional: set up video writer if you want to save the result
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        edges = canny_edge_detector(frame)
        roi = region_of_interest(edges)
        lines = detect_lines(roi)
        lane_lines = average_slope_intercept(frame, lines)
        output_frame = draw_lines(frame, lane_lines)

        if out is not None:
            out.write(output_frame)

        if show:
            cv2.imshow("Lane Detection", output_frame)
            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Put your video name here
    input_video = "input_video.mp4"
    output_video = "output_with_lanes.mp4"

    process_video(input_video, output_path=output_video, show=True)
    print("Processing complete.")
