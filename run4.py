import os
import glob
import cv2
import math
import numpy as np
from kcf import Tracker  # Make sure that the Tracker class from your KCF module is correctly imported

def parse_groundtruth(gt_file):
    """
    Reads the groundtruth file which is assumed to have data in the format: x,y,w,h (comma-separated).
    Any invalid lines are skipped (i.e., lines that cannot be parsed or where w or h <= 0).
    Returns a list of tuples: [(x, y, w, h), ...].
    """
    lines_data = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by comma (the expected format)
            parts = line.split(',')
            if len(parts) != 4:
                continue
            try:
                x, y, w, h = map(float, parts)
                if w <= 0 or h <= 0:
                    continue
                lines_data.append((x, y, w, h))
            except ValueError:
                # If conversion to float fails, skip this line
                continue
    return lines_data

def compute_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Both boxA and boxB should be in the format: (x, y, w, h).
    Returns a float value between 0 and 1.
    """
    Ax1, Ay1, Aw, Ah = boxA
    Bx1, By1, Bw, Bh = boxB

    Ax2 = Ax1 + Aw
    Ay2 = Ay1 + Ah
    Bx2 = Bx1 + Bw
    By2 = By1 + Bh

    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = Aw * Ah
    areaB = Bw * Bh
    iou = inter_area / float(areaA + areaB - inter_area)
    return iou

def compute_success_auc(overlaps, step=0.01):
    """
    Computes the AUC of the Overlap-based Success Plot.
    overlaps: a list of IoU values for each frame.
    step: the step size for the threshold (default 0.01).
    The method calculates, for thresholds t in [0, 1], the percentage of frames where IoU >= t,
    and then averages these success rates to approximate the AUC.
    """
    if not overlaps:
        return 0.0

    thresholds = np.arange(0, 1 + step, step)
    success_rates = []
    for t in thresholds:
        count = sum(1 for ov in overlaps if ov >= t)
        success_rate = count / len(overlaps)
        success_rates.append(success_rate)

    auc = np.mean(success_rates)
    return auc

def run_otb_sequence(seq_folder, dist_threshold=50):
    """
    Runs tracking on a single OTB sequence and calculates the Overlap-based AUC.
    
    Parameters:
      seq_folder: the folder of an OTB sequence (e.g., .../OTB2015/Basketball). This folder is expected
                  to contain an "img" directory with images (e.g., 0001.jpg, 0002.jpg, ...) and a 
                  "groundtruth_rect.txt" file containing ground truth bounding boxes in the format "x,y,w,h"
                  (comma-separated).
      dist_threshold: the threshold (in pixels) for the center distance between the tracked box and the ground truth.
                      If the distance exceeds this value, the tracker is reinitialized using the ground truth.
    
    Returns:
      The Overlap-based AUC for this sequence.
      
    Behavior:
      - The tracker is initialized using the first line of the groundtruth.
      - For each subsequent frame, if the tracking box is invalid or its center deviates from the ground truth 
        by more than the given threshold, a lost flag is set. After the first failure, all subsequent frames'
        IoU are set to 0, and the sequence is terminated.
    """
    img_dir = os.path.join(seq_folder, 'img')
    gt_file = os.path.join(seq_folder, 'groundtruth_rect.txt')

    # Parse the groundtruth data (data is expected to be comma-separated)
    lines_data = parse_groundtruth(gt_file)
    if not lines_data:
        print(f"[Warning] No valid groundtruth data in: {gt_file}. Skipping this sequence.")
        return 0.0

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        print(f"[Warning] No image files found in: {img_dir}")
        return 0.0

    # Initialize tracker with the first groundtruth box
    x, y, w, h = lines_data[0]
    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"[Error] Failed to read first frame: {img_files[0]}")
        return 0.0

    tracker = Tracker()
    tracker.init(first_frame, (x, y, w, h))

    overlaps = []
    lost_flag = False  # Once set to True, all subsequent IoU values are 0
    total_frames = len(img_files)

    for idx, img_path in enumerate(img_files[1:], start=2):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[Warning] Failed to read image: {img_path}")
            break

        px, py, pw, ph = tracker.update(frame)

        if not lost_flag:
            # Check if the tracking box is invalid
            if pw <= 0 or ph <= 0:
                print(f"[Warning] Invalid tracking box at frame {idx}, reinit from groundtruth.")
                if (idx - 1) < len(lines_data):
                    gx, gy, gw, gh = lines_data[idx - 1]
                    tracker = Tracker()
                    tracker.init(frame, (gx, gy, gw, gh))
                    px, py, pw, ph = gx, gy, gw, gh
                lost_flag = True
            else:
                # Otherwise, compare the center distance between the tracked box and groundtruth
                if (idx - 1) < len(lines_data):
                    gx, gy, gw, gh = lines_data[idx - 1]
                else:
                    gx, gy, gw, gh = px, py, pw, ph

                tracker_cx = px + pw / 2.0
                tracker_cy = py + ph / 2.0
                gt_cx = gx + gw / 2.0
                gt_cy = gy + gh / 2.0
                dist = math.sqrt((tracker_cx - gt_cx)**2 + (tracker_cy - gt_cy)**2)

                if dist > dist_threshold:
                    print(f"[Warning] Large deviation at frame {idx}, dist={dist:.2f}, reinit from groundtruth.")
                    if (idx - 1) < len(lines_data):
                        gx, gy, gw, gh = lines_data[idx - 1]
                        tracker = Tracker()
                        tracker.init(frame, (gx, gy, gw, gh))
                        px, py, pw, ph = gx, gy, gw, gh
                    lost_flag = True

        # If lost_flag is set, set all remaining frame overlaps to 0 and exit the loop
        if lost_flag:
            remaining = total_frames - (idx - 1)
            print(f"[Info] Tracking lost at frame {idx}, remaining {remaining} frames set to IoU=0.")
            overlaps.extend([0.0] * remaining)
            break
        else:
            if (idx - 1) < len(lines_data):
                gx, gy, gw, gh = lines_data[idx - 1]
                iou = compute_iou((px, py, pw, ph), (gx, gy, gw, gh))
            else:
                iou = 0.0
            overlaps.append(iou)

        # Convert coordinates to integers for cv2.rectangle
        pt1 = (int(px), int(py))
        pt2 = (int(px + pw), int(py + ph))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame {idx}", (int(px), int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.imshow("KCF Tracking", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cv2.destroyAllWindows()

    seq_auc = compute_success_auc(overlaps, step=0.01)
    print(f"[Info] Sequence AUC for {os.path.basename(seq_folder)}: {seq_auc:.4f}")
    return seq_auc

def run_all_otb_sequences(otb_base, dist_threshold=50):
    """
    Iterates over all subfolders under the specified OTB base directory.
    Each subfolder that contains an "img" directory and a "groundtruth_rect.txt" file is considered a valid sequence.
    The function runs run_otb_sequence() on each valid sequence, collects its AUC, and finally prints the mean AUC.
    
    Parameters:
      otb_base: The root directory of the OTB dataset (e.g., "/path/to/OTB2015/OTB2015").
                **Important:** You must modify this path to the actual path where your OTB dataset is stored.
      dist_threshold: The center distance threshold (in pixels) used for reinitializing the tracker when deviation is too large.
    """
    seq_names = sorted(os.listdir(otb_base))
    all_aucs = []
    valid_count = 0

    for seq_name in seq_names:
        seq_folder = os.path.join(otb_base, seq_name)
        img_dir = os.path.join(seq_folder, 'img')
        gt_file = os.path.join(seq_folder, 'groundtruth_rect.txt')

        if os.path.isdir(seq_folder) and os.path.isdir(img_dir) and os.path.isfile(gt_file):
            print("=" * 60)
            print(f"Running sequence: {seq_name}")
            print("=" * 60)
            seq_auc = run_otb_sequence(seq_folder, dist_threshold=dist_threshold)
            all_aucs.append(seq_auc)
            valid_count += 1
        else:
            print(f"Skipping {seq_name} (not a valid OTB sequence folder).")

    if valid_count > 0:
        mean_auc = sum(all_aucs) / valid_count
        print(f"\n[Result] Mean AUC over {valid_count} valid sequences: {mean_auc:.4f}")
    else:
        print("[Result] No valid sequences processed.")

if __name__ == "__main__":
    # Set the OTB dataset base path to where your OTB2015 data is stored.
    # For example, if your OTB2015 data is located in "/Users/yourname/datasets/OTB2015/OTB2015",
    # update the otb_base variable accordingly.
    otb_base = "../data/OTB2015"
    run_all_otb_sequences(otb_base, dist_threshold=30)