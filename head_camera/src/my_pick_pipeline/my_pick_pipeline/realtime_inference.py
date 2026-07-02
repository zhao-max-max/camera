"""
只训练了红色
两类实时推理 + ROS2 位姿发布（以 realtime_inference.py 为骨架）

模型: best.pt 训练出的两类分割模型
  - red-box  (id=0): 正方体上表面 → 边长 0.25m
  - red-space(id=1): 地面纯色块   → 边长 0.40m

功能与 realtime_inference.py 一致：
  RANSAC 平面拟合 + 可见直角顶点 + 中心估计 + UV轴(相机X轴投影,顺时针最小角)
  + 四元数位姿发布 /head_surface_pose + get_place_pos 服务

区别（保留的新特性）：
  1) 按检测到的类别选掩膜（同帧两类都有时优先 red-box，因为上表面在更上层）
  2) 按命中类别自动切换 expected_side_m（0.25 / 0.40）
"""

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from geometry_msgs.msg import PoseStamped
import math

# 适配现成 C++ 代码里的 robot_msgs
from robot_msgs.srv import GetPlacePos


# ===== 配置 =====
MODEL_PATH = "/home/zyy/task/arm/camera/head_camera/src/best.pt"
CONFIDENCE = 0.45
IOU = 0.5
# DEVICE=0
DEVICE = "cpu"  # 修改为强制使用 cpu 避免 CUDA 报错

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
CAMERA_SERIAL = "043322075459"  # 填入相机序列号；留空则自动选第一个
MASK_THRESH = 0.5

# 类别 -> 物理边长(米)。类别 id 来自模型: {0: 'red-box', 1: 'red-space'}
CLASS_SIDE_M = {
    0: 0.25,   # red-box  : 正方体上表面
    1: 0.40,   # red-space: 地面纯色块
}
CLASS_NAME = {0: "red-box", 1: "red-space"}
# 同一帧同时检出多类时的优先级（数值小者优先）。上表面在最上层，优先它。
CLASS_PRIORITY = {0: 0, 1: 1}

MASK_ALPHA = 0.35
MASK_COLORMAP = cv2.COLORMAP_VIRIDIS

RANSAC_THRESH_M = 0.008
RANSAC_ITERS = 220
MIN_INLIERS = 280
MAX_POINTS_FOR_RANSAC = 25000

POSE_STALE_SEC = 1.0  # 缓存位姿有效期（秒），超过此时间视为过期


def rotmat2quat(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]


class RealSenseCamera:
    def __init__(self, width=CAM_W, height=CAM_H, fps=CAM_FPS, serial=CAMERA_SERIAL):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        color_profile = self.profile.get_stream(rs.stream.color)
        self.intr = color_profile.as_video_stream_profile().get_intrinsics()
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        print("✓ RealSense started")

    def read(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            return None, None
        return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())

    def stop(self):
        self.pipeline.stop()


class PlaneAndCornerEstimator:
    def __init__(self, intr, depth_scale):
        self.intr = intr
        self.depth_scale = depth_scale
        # 当前帧使用的边长，由命中的类别决定（默认 0.25）
        self.expected_side_m = 0.25
        self.prev_corners = []
        self.prev_u_vec = None
        self.prev_v_vec = None

    @staticmethod
    def largest_component(mask_u8: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask_u8, dtype=np.uint8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask_u8, dtype=np.uint8)
        out[labels == largest_label] = 255
        return out

    def extract_mask_points_3d(self, mask_u8: np.ndarray, depth_image: np.ndarray):
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) < 40:
            return None

        if len(xs) > MAX_POINTS_FOR_RANSAC:
            idx = np.linspace(0, len(xs) - 1, MAX_POINTS_FOR_RANSAC).astype(np.int32)
            xs = xs[idx]
            ys = ys[idx]

        d = depth_image[ys, xs].astype(np.float32)
        valid = d > 0
        if np.count_nonzero(valid) < 40:
            return None

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        d = d[valid]

        z = d * self.depth_scale
        x = (xs - self.intr.ppx) * z / self.intr.fx
        y = (ys - self.intr.ppy) * z / self.intr.fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def ransac_plane(pts: np.ndarray):
        n_pts = pts.shape[0]
        if n_pts < 40:
            return None

        best_count = 0
        best_inliers = None
        for _ in range(RANSAC_ITERS):
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts[idx]
            n = np.cross(p2 - p1, p3 - p1)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / (nn + 1e-8)
            d = -float(np.dot(n, p1))
            dist = np.abs(pts @ n + d)
            inliers = dist < RANSAC_THRESH_M
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < MIN_INLIERS:
            return None

        pts_in = pts[best_inliers]
        origin = np.mean(pts_in, axis=0)
        centered = pts_in - origin
        cov = (centered.T @ centered) / max(len(centered), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        if normal[2] < 0:
            normal = -normal
        return origin, normal

    def ray_plane_point(self, px: float, py: float, plane_origin: np.ndarray, plane_normal: np.ndarray):
        vx = (px - self.intr.ppx) / self.intr.fx
        vy = (py - self.intr.ppy) / self.intr.fy
        ray = np.array([vx, vy, 1.0], dtype=np.float64)
        denom = np.dot(plane_normal, ray)
        if abs(denom) < 1e-7:
            return None
        t = np.dot(plane_normal, plane_origin) / denom
        if t <= 0:
            return None
        return t * ray

    def extract_visible_right_angle_vertices(self, mask_u8: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray):
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        h, w = mask_u8.shape[:2]
        border_margin = 20

        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 8:
            return []

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        pts = approx[:, 0, :].astype(np.float64)
        n = len(pts)
        if n < 3:
            return []

        n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
        # 最短可见边阈值随当前类别边长缩放
        min_len_m = max(0.02, self.expected_side_m * 0.10)

        candidates = []
        for i in range(n):
            p_prev_2d = pts[(i - 1 + n) % n]
            p_curr_2d = pts[i]
            p_next_2d = pts[(i + 1) % n]

            p_prev = self.ray_plane_point(p_prev_2d[0], p_prev_2d[1], plane_origin, n_hat)
            p_curr = self.ray_plane_point(p_curr_2d[0], p_curr_2d[1], plane_origin, n_hat)
            p_next = self.ray_plane_point(p_next_2d[0], p_next_2d[1], plane_origin, n_hat)
            if p_prev is None or p_curr is None or p_next is None:
                continue

            e1 = p_prev - p_curr
            e2 = p_next - p_curr
            e1 = e1 - np.dot(e1, n_hat) * n_hat
            e2 = e2 - np.dot(e2, n_hat) * n_hat

            len1 = np.linalg.norm(e1)
            len2 = np.linalg.norm(e2)
            if len1 < min_len_m or len2 < min_len_m:
                continue

            cos_theta = abs(np.dot(e1, e2) / (len1 * len2 + 1e-8))
            if cos_theta < 0.25:  # 约 75°~105°
                score = min(len1, len2) * (1.0 - cos_theta)
                x, y = int(round(p_curr_2d[0])), int(round(p_curr_2d[1]))

                # 处于图像边界附近的点，视为被截断，不算可见角点
                if x <= border_margin or x >= (w - 1 - border_margin) or y <= border_margin or y >= (h - 1 - border_margin):
                    continue

                candidates.append((score, x, y))

        candidates.sort(key=lambda t: t[0], reverse=True)
        corners = []
        for _, x, y in candidates:
            ok = True
            for qx, qy in corners:
                if (x - qx) ** 2 + (y - qy) ** 2 <= 8 * 8:
                    ok = False
                    break
            if ok:
                corners.append((x, y))
            if len(corners) >= 4:
                break

        return corners

    def temporal_stabilize_corners(self, corners):
        if not corners:
            self.prev_corners = []
            return []
        if not self.prev_corners:
            self.prev_corners = list(corners)
            return corners

        prev = list(self.prev_corners)
        curr = list(corners)
        used = set()
        out = []

        for px, py in prev:
            best_j = -1
            best_d2 = 1e18
            for j, (cx, cy) in enumerate(curr):
                if j in used:
                    continue
                d2 = (cx - px) ** 2 + (cy - py) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            if best_j >= 0 and best_d2 < 30 * 30:
                used.add(best_j)
                cx, cy = curr[best_j]
                sx = int(round(0.7 * px + 0.3 * cx))
                sy = int(round(0.7 * py + 0.3 * cy))
                out.append((sx, sy))

        for j, p in enumerate(curr):
            if j not in used:
                out.append(p)

        out = out[:4]
        self.prev_corners = list(out)
        return out

    def project_3d_to_pixel(self, pt3d):
        x, y, z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])
        if z <= 1e-6:
            return None
        px = self.intr.fx * x / z + self.intr.ppx
        py = self.intr.fy * y / z + self.intr.ppy
        return int(round(px)), int(round(py))

    def corners_px_to_3d(self, corners_px, plane_origin, plane_normal):
        pts3d = []
        n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
        for x, y in corners_px:
            p = self.ray_plane_point(float(x), float(y), plane_origin, n_hat)
            if p is not None:
                pts3d.append(np.asarray(p, dtype=np.float64))
        return pts3d

    def estimate_uv_axes(self, corners_3d, plane_normal):
        n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)

        # 默认基向量
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ref, n_hat)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ref = ref - np.dot(ref, n_hat) * n_hat
        ref = ref / (np.linalg.norm(ref) + 1e-8)

        u = None
        if len(corners_3d) >= 2:
            best_pair = None
            best_err = 1e18
            for i in range(len(corners_3d)):
                for j in range(i + 1, len(corners_3d)):
                    e = corners_3d[j] - corners_3d[i]
                    e = e - np.dot(e, n_hat) * n_hat
                    le = np.linalg.norm(e)
                    if le < 1e-4:
                        continue
                    # 误差参考当前类别边长
                    err = abs(le - self.expected_side_m)
                    if err < best_err:
                        best_err = err
                        best_pair = e

            if best_pair is not None:
                u = best_pair / (np.linalg.norm(best_pair) + 1e-8)

        if u is None:
            u = self.prev_u_vec if self.prev_u_vec is not None else ref

        u = u - np.dot(u, n_hat) * n_hat
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(n_hat, u)
        v = v / (np.linalg.norm(v) + 1e-8)

        # ===== 基于相机X轴投影寻找顺时针夹角最小方向作为 U 轴 =====
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_proj = x_axis - np.dot(x_axis, n_hat) * n_hat
        x_proj_norm = np.linalg.norm(x_proj)

        if x_proj_norm > 1e-6:
            x_proj = x_proj / x_proj_norm
            # 定义平面内顺时针的正交方向 cw_dir。
            # 图像坐标系中X向右，Y向下，Z向内。法向 n_hat 默认Z>0。
            # 在右手坐标系中，n_hat × x_proj 会得到近似 Y 轴（向下，也就是图像上的顺时针方向）的方向
            cw_dir = np.cross(n_hat, x_proj)
            cw_dir = cw_dir / (np.linalg.norm(cw_dir) + 1e-8)

            opts = [u, -u, v, -v]
            best_u = None
            min_angle = float('inf')

            for opt in opts:
                # 计算 opt 在该平面内的分量
                val_y = np.dot(opt, x_proj)
                val_x = np.dot(opt, cw_dir)

                # arctan2(val_x, val_y) 求出相对于 x_proj 以 cw_dir 为正方向（顺时针）的夹角
                angle_cw = np.arctan2(val_x, val_y)
                if angle_cw < 0:
                    angle_cw += 2 * np.pi

                if angle_cw < min_angle:
                    min_angle = angle_cw
                    best_u = opt

            u = best_u
            # 根据法向量倒推求得顺应右手系的 V 轴
            v = np.cross(n_hat, u)
            v = v / (np.linalg.norm(v) + 1e-8)

        # 低速平滑以防单帧计算抖动 (只有当U轴未发生90度跳变时才平滑)
        if self.prev_u_vec is not None and np.dot(u, self.prev_u_vec) > 0.5:
            alpha = 0.6
            u = (1.0 - alpha) * self.prev_u_vec + alpha * u
            u = u - np.dot(u, n_hat) * n_hat
            u = u / (np.linalg.norm(u) + 1e-8)
            v = np.cross(n_hat, u)
            v = v / (np.linalg.norm(v) + 1e-8)

        self.prev_u_vec = u
        self.prev_v_vec = v
        return u, v

    def compute_center_from_corner_count(self, corners_3d, plane_origin, axis_u, axis_v, mask_centroid_uv=None,
                                         mask_u8=None, plane_normal=None):
        if len(corners_3d) == 0:
            return None, None

        uv_pts = []
        for p in corners_3d:
            d = p - plane_origin
            uv_pts.append(np.array([np.dot(d, axis_u), np.dot(d, axis_v)], dtype=np.float64))

        n = len(uv_pts)
        if n >= 4:
            center_uv = np.mean(np.array(uv_pts[:4]), axis=0)
        elif n == 3:
            # 三点补全：最长边为对角线
            d01 = np.linalg.norm(uv_pts[0] - uv_pts[1])
            d12 = np.linalg.norm(uv_pts[1] - uv_pts[2])
            d20 = np.linalg.norm(uv_pts[2] - uv_pts[0])
            if d01 >= d12 and d01 >= d20:
                A, B, C = uv_pts[0], uv_pts[1], uv_pts[2]
            elif d12 >= d01 and d12 >= d20:
                A, B, C = uv_pts[1], uv_pts[2], uv_pts[0]
            else:
                A, B, C = uv_pts[2], uv_pts[0], uv_pts[1]
            D = A + B - C
            center_uv = (A + B + C + D) / 4.0
        elif n == 2:
            # 约束：两点只按"相邻边点"处理（不存在对角点情况）
            A, B = uv_pts[0], uv_pts[1]
            mid = 0.5 * (A + B)
            t = B - A
            nt = np.linalg.norm(t)
            if nt < 1e-8:
                center_uv = mid
            else:
                t = t / nt
                n2 = np.array([-t[1], t[0]], dtype=np.float64)
                c1 = mid + 0.5 * self.expected_side_m * n2
                c2 = mid - 0.5 * self.expected_side_m * n2

                # ===== 旧逻辑：仅用可见掩膜质心距离消歧（质心贴边/出框时不可靠，已注释）=====
                # if mask_centroid_uv is not None:
                #     # 仅使用当前帧掩膜质心消歧：选离质心更近的候选
                #     d1 = np.linalg.norm(c1 - mask_centroid_uv)
                #     d2 = np.linalg.norm(c2 - mask_centroid_uv)
                #     center_uv = c1 if d1 <= d2 else c2
                # else:
                #     center_uv = c1 if np.linalg.norm(c1) <= np.linalg.norm(c2) else c2

                # ===== 新逻辑：掩膜多数像素投票决定 AB 哪一侧是中心侧 =====
                # 可见上表面一定铺在 AB 朝向中心的那一侧。统计掩膜像素相对 mid
                # 沿 n2 的符号，多数票决定方向。即使真实中心出框、只剩贴 AB 的窄带，
                # 这条窄带也 100% 在中心侧，投票干净。
                side = 0.0
                if mask_u8 is not None and plane_normal is not None:
                    ys_m, xs_m = np.where(mask_u8 > 0)
                    stride = max(1, len(xs_m) // 2000)  # 控制采样量
                    acc = 0.0
                    for mx, my in zip(xs_m[::stride], ys_m[::stride]):
                        p = self.ray_plane_point(float(mx), float(my), plane_origin, plane_normal)
                        if p is None:
                            continue
                        d = p - plane_origin
                        uv = np.array([np.dot(d, axis_u), np.dot(d, axis_v)], dtype=np.float64)
                        acc += np.sign(np.dot(uv - mid, n2))
                    side = acc

                if side > 0:
                    center_uv = c1            # 多数像素在 n2 正向 → c1 侧
                elif side < 0:
                    center_uv = c2
                elif mask_centroid_uv is not None:
                    # 投票打平/无掩膜信息：退回质心距离
                    d1 = np.linalg.norm(c1 - mask_centroid_uv)
                    d2 = np.linalg.norm(c2 - mask_centroid_uv)
                    center_uv = c1 if d1 <= d2 else c2
                else:
                    center_uv = c1 if np.linalg.norm(c1) <= np.linalg.norm(c2) else c2
        else:  # n == 1
            # 单角点：从角点分别沿 U/V 方向偏移半边长，四种组合里选最合理中心
            corner = uv_pts[0]
            h = 0.5 * self.expected_side_m
            cands = [
                corner + np.array([h, h], dtype=np.float64),
                corner + np.array([h, -h], dtype=np.float64),
                corner + np.array([-h, h], dtype=np.float64),
                corner + np.array([-h, -h], dtype=np.float64),
            ]
            if mask_centroid_uv is not None:
                # 仅使用当前帧掩膜质心消歧
                dists = [np.linalg.norm(c - mask_centroid_uv) for c in cands]
            else:
                dists = [np.linalg.norm(c) for c in cands]
            center_uv = cands[int(np.argmin(dists))]

        center_3d = plane_origin + center_uv[0] * axis_u + center_uv[1] * axis_v
        return center_3d, center_uv

    @staticmethod
    def draw_mask_overlay(image: np.ndarray, mask_u8: np.ndarray):
        out = image.copy()
        mask_bool = mask_u8 > 0
        if np.any(mask_bool):
            mask_col = cv2.applyColorMap(mask_u8, MASK_COLORMAP)
            blend = cv2.addWeighted(out, 1 - MASK_ALPHA, mask_col, MASK_ALPHA, 0)
            out[mask_bool] = blend[mask_bool]
        return out

    @staticmethod
    def draw_corners(image: np.ndarray, corners, label_text=""):
        out = image.copy()
        for x, y in corners:
            cv2.circle(out, (x, y), 5, (255, 255, 0), -1)
            cv2.circle(out, (x, y), 9, (0, 128, 255), 2)
        txt = f"visible right-angle corners: {len(corners)}"
        if label_text:
            txt = f"[{label_text}] " + txt
        cv2.putText(out, txt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    def draw_uv_axes_and_center(self, image: np.ndarray, center_3d, axis_u, axis_v, normal=None, axis_len_m=0.08, min_angle=None):
        out = image.copy()
        c_px = self.project_3d_to_pixel(center_3d)
        if c_px is None:
            return out

        u_px = self.project_3d_to_pixel(center_3d + axis_len_m * axis_u)
        v_px = self.project_3d_to_pixel(center_3d + axis_len_m * axis_v)

        cv2.circle(out, c_px, 6, (0, 0, 255), -1)

        text = "Center"
        if min_angle is not None:
            text += f" (Min Angle: {min_angle:.1f} deg)"
        cv2.putText(out, text, (c_px[0] + 8, c_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if u_px is not None:
            cv2.arrowedLine(out, c_px, u_px, (0, 0, 255), 2, tipLength=0.2)
            cv2.putText(out, "U", (u_px[0] + 5, u_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if v_px is not None:
            cv2.arrowedLine(out, c_px, v_px, (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(out, "V", (v_px[0] + 5, v_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制投影到平面的相机X轴
        if normal is not None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n_hat = normal / (np.linalg.norm(normal) + 1e-8)
            x_proj = x_axis - np.dot(x_axis, n_hat) * n_hat
            x_proj_norm = np.linalg.norm(x_proj)
            if x_proj_norm > 1e-6:
                x_proj = x_proj / x_proj_norm
                x_px = self.project_3d_to_pixel(center_3d + axis_len_m * x_proj)
                if x_px is not None:
                    cv2.arrowedLine(out, c_px, x_px, (255, 255, 0), 2, tipLength=0.2)
                    cv2.putText(out, "X_proj", (x_px[0] + 5, x_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return out


def select_mask_by_class(result, thresh=MASK_THRESH):
    """
    按类别选取掩膜：
      - 同帧检出多类时，按 CLASS_PRIORITY 选优先级最高的类别（red-box 优先）
      - 同类多个实例时，合并后取最大连通域
    返回: (mask_u8, class_id) 或 (None, None)
    """
    if result.masks is None or len(result.masks) == 0:
        return None, None
    if result.boxes is None or result.boxes.cls is None:
        return None, None

    masks = result.masks.data.cpu().numpy()               # (N, h, w)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)   # (N,)

    # 在检出的类别里挑优先级最高的
    present = sorted(set(cls_ids.tolist()), key=lambda c: CLASS_PRIORITY.get(c, 99))
    if not present:
        return None, None
    chosen_cls = present[0]

    sel = cls_ids == chosen_cls
    chosen_masks = masks[sel]
    if chosen_masks.shape[0] == 0:
        return None, None

    merged = (np.any(chosen_masks > thresh, axis=0).astype(np.uint8) * 255)
    mask_u8 = PlaneAndCornerEstimator.largest_component(merged)
    return mask_u8, chosen_cls


def main(args=None):
    if not os.path.exists(MODEL_PATH):
        print(f"✗ model not found: {MODEL_PATH}")
        return

    # ===== ROS 2 节点初始化 =====
    rclpy.init(args=args)
    node = rclpy.create_node('head_surface_center_estimator')
    pose_pub = node.create_publisher(PoseStamped, '/head_surface_pose', 10)

    # 缓存位姿状态，供服务回调使用
    global_state = {
        'pose': None,
        'last_success_time': 0.0,  # time.monotonic() 时间戳
    }

    # 定义服务回调函数
    def handle_get_place_pose(request, response):
        age = time.monotonic() - global_state['last_success_time']
        if global_state['pose'] is not None and age < POSE_STALE_SEC:
            response.place_pose = global_state['pose']
            response.success = True
            node.get_logger().info(
                f'Service called for {request.frame_name}: returned cached pose (age={age:.2f}s).')
        else:
            response.place_pose = PoseStamped()
            response.success = False
            node.get_logger().warn(
                f'Service called for {request.frame_name}: no valid pose (age={age:.2f}s).')
        return response

    srv = node.create_service(GetPlacePos, 'get_place_pos', handle_get_place_pose)

    cam = RealSenseCamera()
    model = YOLO(MODEL_PATH)
    estimator = PlaneAndCornerEstimator(cam.intr, cam.depth_scale)

    print("\nTwo-class mode (red-box / red-space) + ROS2. Press q to quit.")

    try:
        while rclpy.ok():
            # 处理 ROS 2 回调（响应外界对该服务的请求）
            rclpy.spin_once(node, timeout_sec=0.005)

            color, depth = cam.read()
            if color is None:
                continue

            result = model.predict(source=color, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]
            mask_u8, cls_id = select_mask_by_class(result, MASK_THRESH)

            vis = color.copy()
            corners = []
            center_3d = None
            axis_u, axis_v = None, None
            cls_label = ""
            if mask_u8 is not None:
                # 关键：按命中类别切换物理边长
                estimator.expected_side_m = CLASS_SIDE_M.get(cls_id, 0.25)
                cls_label = f"{CLASS_NAME.get(cls_id, '?')} side={estimator.expected_side_m:.2f}m"

                pts3d = estimator.extract_mask_points_3d(mask_u8, depth)
                if pts3d is not None:
                    plane = estimator.ransac_plane(pts3d)
                    if plane is not None:
                        origin, normal = plane
                        corners = estimator.extract_visible_right_angle_vertices(mask_u8, origin, normal)
                        corners = estimator.temporal_stabilize_corners(corners)

                        corners_3d = estimator.corners_px_to_3d(corners, origin, normal)
                        axis_u, axis_v = estimator.estimate_uv_axes(corners_3d, normal)

                        # 计算相机X轴(1, 0, 0)与 U、V 轴的锐角夹角
                        x_axis_ref = np.array([1.0, 0.0, 0.0])
                        angle_u = np.degrees(np.arccos(np.clip(abs(np.dot(axis_u, x_axis_ref)), 0.0, 1.0)))
                        angle_v = np.degrees(np.arccos(np.clip(abs(np.dot(axis_v, x_axis_ref)), 0.0, 1.0)))
                        min_angle = min(angle_u, angle_v)

                        # 计算当前帧掩膜质心并投影到 UV 平面，用于中心消歧
                        ys_m, xs_m = np.where(mask_u8 > 0)
                        cx_px = float(np.mean(xs_m))
                        cy_px = float(np.mean(ys_m))
                        centroid_3d = estimator.ray_plane_point(cx_px, cy_px, origin, normal)
                        if centroid_3d is not None:
                            dc = centroid_3d - origin
                            mask_centroid_uv = np.array([np.dot(dc, axis_u), np.dot(dc, axis_v)], dtype=np.float64)
                        else:
                            mask_centroid_uv = None

                        center_3d, _ = estimator.compute_center_from_corner_count(corners_3d, origin, axis_u, axis_v, mask_centroid_uv, mask_u8=mask_u8, plane_normal=normal)
                        if center_3d is not None:
                            print(f"[{CLASS_NAME.get(cls_id, '?')}] Center3D: X={center_3d[0]:.4f}  Y={center_3d[1]:.4f}  Z={center_3d[2]:.4f}  side={estimator.expected_side_m:.2f}m  visible_corners={len(corners_3d)}  min_angle={min_angle:.1f}°")

                            # 基于 U、V 和平面法向量构建 3x3 旋转矩阵
                            # X轴=U方向，Y轴=V方向，Z轴=平面法向量(指向相机的反向)
                            n_hat = normal / (np.linalg.norm(normal) + 1e-8)
                            R = np.column_stack((axis_u, axis_v, n_hat))
                            q = rotmat2quat(R)

                            pose_msg = PoseStamped()
                            pose_msg.header.stamp = node.get_clock().now().to_msg()
                            pose_msg.header.frame_id = "dog_camera_link"

                            pose_msg.pose.position.x = float(center_3d[0])
                            pose_msg.pose.position.y = float(center_3d[1])
                            pose_msg.pose.position.z = float(center_3d[2])

                            pose_msg.pose.orientation.x = float(q[0])
                            pose_msg.pose.orientation.y = float(q[1])
                            pose_msg.pose.orientation.z = float(q[2])
                            pose_msg.pose.orientation.w = float(q[3])

                            pose_pub.publish(pose_msg)

                            # 更新全局缓存，供服务直接返回同一份结果
                            global_state['pose'] = pose_msg
                            global_state['last_success_time'] = time.monotonic()

                vis = estimator.draw_mask_overlay(vis, mask_u8)
                vis = estimator.draw_corners(vis, corners, label_text=cls_label)

                # 放到最后画，确保中心点和UV轴不会被掩膜覆盖
                if center_3d is not None and axis_u is not None and axis_v is not None:
                    vis = estimator.draw_uv_axes_and_center(vis, center_3d, axis_u, axis_v, normal=normal, min_angle=min_angle)

            cv2.imshow("head_camera: Two-class Mask Visible Right-Angle Corners", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
