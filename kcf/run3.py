import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kcf import Tracker

def run_otb_sequence(seq_folder, visualize=False):
    """
    对单个OTB序列进行跟踪:
      seq_folder: 目录下应含 `img/` (若干.jpg帧) 与 `groundtruth_rect.txt` (x,y,w,h)
    返回:
      tracking_results, ground_truth_list
       - tracking_results: [(px,py,pw,ph), ...] 每帧跟踪输出
       - ground_truth_list: [(gx,gy,gw,gh), ...] 对应GT
    """
    img_dir = os.path.join(seq_folder, 'img')
    gt_file = os.path.join(seq_folder, 'groundtruth_rect.txt')

    # 读取所有图像帧
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        print(f"[WARNING] 没有找到图像文件: {img_dir}")
        return [], []

    # 读取GT
    with open(gt_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    ground_truth_list = []
    for l in lines:
        # 兼容不同分隔符
        vals = l.replace('\t', ',').replace(' ', ',').split(',')
        x, y, w, h = [int(float(v)) for v in vals[:4]]
        ground_truth_list.append((x,y,w,h))

    num_frames = min(len(img_files), len(ground_truth_list))
    if num_frames == 0:
        print(f"[WARNING] 图像数/GT数=0: {seq_folder}")
        return [], []

    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"[ERROR] 第一帧图像读取失败: {img_files[0]}")
        return [], []

    # 初始化 KCF
    tracker = Tracker()
    x0, y0, w0, h0 = ground_truth_list[0]
    tracker.init(first_frame, (x0,y0,w0,h0))

    tracking_results = []
    tracking_results.append((x0,y0,w0,h0))  # 第一帧的结果

    # 跟踪后续帧
    for idx in range(1, num_frames):
        frame = cv2.imread(img_files[idx])
        if frame is None:
            break

        px, py, pw, ph = tracker.update(frame)
        tracking_results.append((px,py,pw,ph))

        if visualize:
            # 可视化: 画框, 显示
            cv2.rectangle(frame, (px,py), (px+pw,py+ph), (0,255,255), 2)
            cv2.putText(frame, f"Frame {idx+1}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("KCF", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    if visualize:
        cv2.destroyAllWindows()

    # 截断 ground_truth_list 到跟踪帧数
    ground_truth_list = ground_truth_list[:len(tracking_results)]
    return tracking_results, ground_truth_list


def iou(boxA, boxB):
    """
    计算两矩形(A, B)的IoU: box=(x,y,w,h) 左上角 + 宽高
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    Ax2 = xA + wA
    Ay2 = yA + hA
    Bx2 = xB + wB
    By2 = yB + hB

    interX1 = max(xA, xB)
    interY1 = max(yA, yB)
    interX2 = min(Ax2, Bx2)
    interY2 = min(Ay2, By2)
    interW = max(0, interX2 - interX1)
    interH = max(0, interY2 - interY1)
    interArea = interW * interH

    areaA = wA * hA
    areaB = wB * hB
    unionArea = areaA + areaB - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / float(unionArea)


def compute_overlaps(tracking_results, ground_truth_list):
    overlaps = []
    for trk, gt in zip(tracking_results, ground_truth_list):
        overlaps.append(iou(trk, gt))
    return overlaps


def success_plot(overlaps):
    """
    overlaps: 每帧的 IoU
    return: thresholds(0~1), success_rates
    """
    import numpy as np
    thresholds = np.linspace(0,1,101)
    success_rates = []
    for t in thresholds:
        rate = np.mean([o>=t for o in overlaps])
        success_rates.append(rate)
    return thresholds, success_rates

def auc_of_success_plot(success_rates):
    """
    对 success_rates做平均，相当于数值积分
    """
    return np.mean(success_rates)


def main():
    otb_base = "/Users/luowanju/kcf/KCF/OTB2015/OTB2015"

    # 列出子文件夹并排序(让它按字母顺序来)
    all_dirs = sorted(os.listdir(otb_base))

    # 存放总体 Overlap
    all_overlaps = []

    # 一个一个地处理
    for d in all_dirs:
        seq_path = os.path.join(otb_base, d)
        if not os.path.isdir(seq_path):
            continue

        # 确认此序列包含 groundtruth_rect.txt 与 img/
        if not os.path.isfile(os.path.join(seq_path,"groundtruth_rect.txt")):
            continue
        if not os.path.isdir(os.path.join(seq_path,"img")):
            continue

        print(f"\n===== 开始处理序列: {d} =====")
        tracking_res, gt_list = run_otb_sequence(seq_path, visualize=False)
        if len(tracking_res)==0 or len(gt_list)==0:
            print(f"  -> 无法运行或无结果，跳过.")
            continue

        # 计算本序列 Overlaps + AUC
        seq_overlaps = compute_overlaps(tracking_res, gt_list)
        seq_thresholds, seq_succ = success_plot(seq_overlaps)
        seq_auc = auc_of_success_plot(seq_succ)

        print(f"  -> 序列 {d} 的帧数: {len(seq_overlaps)}; AUC={seq_auc:.3f}")

        # 拼到全局
        all_overlaps.extend(seq_overlaps)

    # 计算整体 AUC
    if not all_overlaps:
        print("\n[总结果] 没有任何成功的overlaps，无法计算AUC.")
        return
    thresholds, success_rates = success_plot(all_overlaps)
    auc_val = auc_of_success_plot(success_rates)
    print(f"\n[总结果] 所有序列汇总 AUC: {auc_val:.3f}")

    # 如需可视化
    plt.figure()
    plt.plot(thresholds, success_rates, label=f"Overall AUC={auc_val:.3f}")
    plt.xlabel("Overlap Threshold")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.title("OTB2015 - KCF Overall Success Plot")
    plt.show()


if __name__=="__main__":
    main()