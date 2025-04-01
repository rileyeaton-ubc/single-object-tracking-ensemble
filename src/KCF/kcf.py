import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real

class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding=(0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h * grid, w * grid))
        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(np.pi / 9 * k))
                    cv2.rectangle(img, (j * grid, i * grid),
                                  ((j + 1) * grid, (i + 1) * grid),
                                  (255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)

class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

    def get_feature(self, image, roi):
        # roi 格式为 (cx, cy, w, h)
        cx, cy, w, h = roi
        # 根据 padding 调整目标区域尺寸，确保宽高为偶数
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        # 边界检查：确保 ROI 在图像内部
        img_h, img_w = image.shape[:2]
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        if x1 >= x2 or y1 >= y2:
            print("Warning: ROI is outside the image. Returning default zero feature.")
            return self._get_zero_feature()

        sub_image = image[y1:y2, x1:x2, :]
        if sub_image.size == 0:
            print("Warning: sub_image is empty after bounds check. Returning default zero feature.")
            return self._get_zero_feature()

        # 调整子区域大小
        resized_image = cv2.resize(sub_image, (self.pw, self.ph))
        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_image)
            if self.debug:
                self.hog.show_hog(feature)

        fc, fh_feat, fw_feat = feature.shape
        self.scale_h = float(fh_feat) / h if h != 0 else 1.0
        self.scale_w = float(fw_feat) / w if w != 0 else 1.0

        # 添加汉宁窗
        hann2t, hann1t = np.ogrid[0:fh_feat, 0:fw_feat]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw_feat - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh_feat - 1)))
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def _get_zero_feature(self):
        """
        返回默认的零特征，防止 ROI 无效时程序崩溃。
        根据是否使用灰度特征返回不同尺寸的零数组。
        """
        if self.gray_feature:
            dummy = np.zeros((1, self.ph, self.pw), dtype=np.float32)
            return dummy
        else:
            # 对于 HOG 特征，返回与 HOG.get_feature() 预期输出相似的维度。
            sw, sh = self.hog.blockStride
            w_new = self.pw // sw - 1
            h_new = self.ph // sh - 1
            dummy = np.zeros((36, h_new, w_new), dtype=np.float32)
            return dummy

    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def train(self, x, y, sigma, lambdar):
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, alphaf, x, z, sigma):
        k = self.kernel_correlation(x, z, sigma)
        return real(ifft2(self.alphaf * fft2(k)))

    def kernel_correlation(self, x1, x2, sigma):
        c = ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def init(self, image, roi):
        # roi 格式为 (x, y, w, h)，初始化时转换为 (cx, cy, w, h)
        x1, y1, w, h = roi     
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        x = self.get_feature(image, roi)
        y_gaussian = self.gaussian_peak(x.shape[2], x.shape[1])
        self.alphaf = self.train(x, y_gaussian, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        cx, cy, w, h = self.roi
        max_response = -1
        best_dx, best_dy, best_w, best_h, best_z = 0, 0, 0, 0, None
        for scale in [0.95, 1.0, 1.05]:
            # 计算新的 roi，注意 map(int, ...) 返回迭代器，这里转换为 tuple
            roi = tuple(map(int, (cx, cy, w * scale, h * scale)))
            z = self.get_feature(image, roi)
            responses = self.detect(self.alphaf, self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                best_dx = int((idx % width - width / 2) / self.scale_w)
                best_dy = int((idx // width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        cx_new = cx + best_dx
        cy_new = cy + best_dy
        self.roi = (cx_new, cy_new, best_w, best_h)
        # 更新模板
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y_gaussian = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y_gaussian, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        return (cx_new - best_w // 2, cy_new - best_h // 2, best_w, best_h)