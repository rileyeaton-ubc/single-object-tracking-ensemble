import os
import glob
import cv2
import math
import numpy as np

from kcf import Tracker  # 确保能正确导入你的 KCF Tracker 类

def fix_groundtruth(gt_file):
    """
    读取并修正 groundtruth 文件：
    1) 将逗号替换为空格；
    2) 过滤掉无法解析或 w,h<=0 的行；
    3) 最终以 "x y w h" 格式写回同一个文件。
    注意：会直接覆盖原 groundtruth 文件，先备份！
    """
    lines_fixed = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 将逗号替换为空格
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                x, y, w, h = map(int, map(float, parts))
                if w <= 0 or h <= 0:
                    continue
                line_fixed = f"{x} {y} {w} {h}\n"
                lines_fixed.append(line_fixed)
            except ValueError:
                continue

    with open(gt_file, 'w') as f:
        f.writelines(lines_fixed)

    print(f"[fix_groundtruth] {gt_file} 修正完成，共保留 {len(lines_fixed)} 行有效数据。")


def compute_iou(boxA, boxB):
    """
    计算两个边界框的 IoU (Intersection over Union)
    boxA, boxB 格式: (x, y, w, h)
    返回值: IoU 浮点数, 0 ~ 1
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
    计算 Overlap-based Success Plot 的 AUC。
    overlaps: 每帧的 IoU 列表
    step: 阈值步长, 默认为 0.01
    原理：对阈值 t 从 0 ~ 1，统计 IoU >= t 的帧比例，然后对这些比例求平均。
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
    seq_folder: OTB 序列的目录，比如 .../OTB2015/Basketball
      里面应有：
        - img/ (包含 0001.jpg, 0002.jpg, ...)
        - groundtruth_rect.txt (每一行格式为 x,y,w,h)
    dist_threshold: 中心点距离阈值，超过这个距离就重新初始化 (用于纠正漂移)
    返回值：该序列的 Overlap-based AUC

    当第一次检测到跟踪失败（无效框或偏差过大）时，将 lost_flag 置为 True，
    后续帧的 IoU 均记为 0，并自动计算剩余帧数补零，然后跳出该序列循环。
    """
    img_dir = os.path.join(seq_folder, 'img')
    gt_file = os.path.join(seq_folder, 'groundtruth_rect.txt')

    fix_groundtruth(gt_file)

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        print(f"[Warning] No image files found in: {img_dir}")
        return 0.0

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        print(f"[Warning] No valid groundtruth data in: {gt_file}")
        return 0.0

    # 用第一行做初始化
    x, y, w, h = [int(float(v)) for v in lines[0].strip().split()]
    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"[Error] Failed to read first frame: {img_files[0]}")
        return 0.0

    tracker = Tracker()
    tracker.init(first_frame, (x, y, w, h))

    overlaps = []
    lost_flag = False  # 一旦置 True，后续 IoU 全记为 0

    total_frames = len(img_files)  # 序列总帧数

    for idx, img_path in enumerate(img_files[1:], start=2):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[Warning] Failed to read image: {img_path}")
            break

        px, py, pw, ph = tracker.update(frame)

        if not lost_flag:
            if pw <= 0 or ph <= 0:
                print(f"[Warning] Invalid tracking box at frame {idx}, reinit from groundtruth.")
                if idx - 1 < len(lines):
                    gx, gy, gw, gh = [int(float(v)) for v in lines[idx - 1].strip().split()]
                    tracker = Tracker()
                    tracker.init(frame, (gx, gy, gw, gh))
                    px, py, pw, ph = gx, gy, gw, gh
                lost_flag = True
            else:
                if idx - 1 < len(lines):
                    gx, gy, gw, gh = [int(float(v)) for v in lines[idx - 1].strip().split()]
                else:
                    gx, gy, gw, gh = px, py, pw, ph

                tracker_cx = px + pw / 2.0
                tracker_cy = py + ph / 2.0
                gt_cx = gx + gw / 2.0
                gt_cy = gy + gh / 2.0
                dist = math.sqrt((tracker_cx - gt_cx)**2 + (tracker_cy - gt_cy)**2)

                if dist > dist_threshold:
                    print(f"[Warning] Large deviation at frame {idx}, dist={dist:.2f}, reinit from groundtruth.")
                    tracker = Tracker()
                    tracker.init(frame, (gx, gy, gw, gh))
                    px, py, pw, ph = gx, gy, gw, gh
                    lost_flag = True

        # 如果已经丢失，则补充剩余帧的 IoU 为 0，并退出循环
        if lost_flag:
            remaining = total_frames - (idx - 1)
            print(f"[Info] Tracking lost at frame {idx}, remaining {remaining} frames set to IoU=0.")
            overlaps.extend([0.0] * remaining)
            break
        else:
            if idx - 1 < len(lines):
                gx, gy, gw, gh = [int(float(v)) for v in lines[idx - 1].strip().split()]
                iou = compute_iou((px, py, pw, ph), (gx, gy, gw, gh))
            else:
                iou = 0.0
            overlaps.append(iou)

        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        cv2.putText(frame, f"Frame {idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("KCF Tracking", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cv2.destroyAllWindows()

    seq_auc = compute_success_auc(overlaps, step=0.01)
    print(f"[Info] Sequence AUC for {os.path.basename(seq_folder)}: {seq_auc:.4f}")
    return seq_auc


def run_all_otb_sequences(otb_base, dist_threshold=50):
    """
    遍历 otb_base 下所有子文件夹，每个包含 img/ 与 groundtruth_rect.txt 的，
    视为有效序列，调用 run_otb_sequence() 并统计其 AUC，
    最后输出所有序列的平均 AUC。
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
    otb_base = "/Users/luowanju/kcf/KCF/OTB2015/OTB2015"
    run_all_otb_sequences(otb_base, dist_threshold=30)