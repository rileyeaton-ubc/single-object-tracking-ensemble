import os
import glob
import cv2
from kcf import Tracker  # 这里要确保能正确导入你的 KCF Tracker 类

def run_otb_sequence(seq_folder):
    """
    seq_folder: OTB 序列的目录，比如  .../OTB2015/Basketball
      里面应有：
        - img/  (包含 0001.jpg, 0002.jpg, ...)
        - groundtruth_rect.txt (第一行是 x,y,w,h)
    """
    img_dir = os.path.join(seq_folder, 'img')
    gt_file = os.path.join(seq_folder, 'groundtruth_rect.txt')

    # 读取所有帧的文件名并排序
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        print(f"没有找到图像文件: {img_dir}")
        return

    # 读取 ground truth: 先拿第一帧的 (x,y,w,h) 做初始化
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    # 以 OTB 通用格式“x,y,w,h”解析第一行：
    x, y, w, h = [int(float(v)) for v in lines[0].strip().split(',')]

    # 读取第一帧图像
    first_frame = cv2.imread(img_files[0])
    if first_frame is None:
        print(f"第一帧图像读取失败: {img_files[0]}")
        return

    # 初始化 KCF 跟踪器
    tracker = Tracker()
    tracker.init(first_frame, (x, y, w, h))

    # 循环处理后续帧
    count = 1
    for idx, img_path in enumerate(img_files[1:], start=2):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"图像读取失败: {img_path}")
            break

        # 跟踪更新
        try:
            px, py, pw, ph = tracker.update(frame)
        except Exception as e:
            count += 1
            print(f"Error occured:{e.with_traceback}")
            print(f"Frame {idx} failed to detect")
            print(f"Total failed detection: {count}")
            px, py, pw, ph = [int(float(v)) for v in lines[idx].strip().split(',')]
        # 画出跟踪框做简单可视化
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        cv2.putText(frame, f"Frame {idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2)
        cv2.imshow("KCF Tracking", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 假设你想跑“Basketball”序列
    otb_base = "/Users/luowanju/kcf/KCF/OTB2015/OTB2015"  # OTB 数据集根目录
    seq_name = "Biker"
    seq_folder = os.path.join(otb_base, seq_name) 

    run_otb_sequence(seq_folder)