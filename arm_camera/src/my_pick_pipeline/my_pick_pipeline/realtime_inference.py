"""
只做一个功能：
基于 YOLO 掩膜，标记“未遮挡的上表面直角顶点”。

增强版要点：
1) 用深度点做 RANSAC 平面拟合，减少纯2D透视误判
2) 角点评分在3D平面内进行（近似90° + 边长支撑）
3) 加简单时序匹配，降低顶点跳动
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

from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose

# 适配现成 C++ 代码里的 robot_msgs
from robot_msgs.srv import GetPickPos


# ===== 配置 =====
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "best6.27.pt"))
CONFIDENCE = 0.45
IOU = 0.5
# DEVICE=0
DEVICE = "cpu"  # 修改为强制使用 cpu 避免 CUDA 报错

CAM_W, CAM_H, CAM_FPS = 640, 480, 30
CAMERA_SERIAL = os.environ.get("CAMERA_SERIAL", "")  # 由 run_neweyes.sh --1/--2 传入
MASK_THRESH = 0.5
EXPECTED_SIDE_M = 0.25

MASK_ALPHA = 0.35
MASK_COLORMAP = cv2.COLORMAP_VIRIDIS

RANSAC_THRESH_M = 0.008
RANSAC_ITERS = 220
MIN_INLIERS = 280
MAX_POINTS_FOR_RANSAC = 25000

POSE_STALE_SEC = 1.0  # 缓存位姿有效期（秒），超过此时间视为过期

# ===== TF 变换配置 =====
# 相机端直接把检测位姿变换到 world 系再缓存/发布。
# 物体在 world 系静止，检测瞬间冻结的 world 坐标不随相机后续移动失效。
TARGET_FRAME = "world"          # 目标坐标系（arm 的 world）
SOURCE_FRAME = "camera_link"    # 相机检测结果所在坐标系
TF_TIMEOUT_SEC = 0.2            # lookup_transform 等待超时

def rotmat2quat(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr+1.0) * 2.0
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

    @staticmethod
    def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
        """填补掩膜内部封闭空洞。

        用漫水法从边界外的背景开始填充：能被外部背景连通到的才是“真背景”，
        剩下没被连通到的黑洞即为内部空洞，将其并回前景。
        """
        if not np.any(mask_u8):
            return mask_u8
        h, w = mask_u8.shape[:2]
        # floodFill 需要比图像大一圈的掩码
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flood = mask_u8.copy()
        # 从 (0,0) 背景点漫水，标记所有与外部连通的背景
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        # flood 中仍为 0 的像素 = 内部空洞
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(mask_u8, holes)

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
        min_len_m = max(0.02, EXPECTED_SIDE_M * 0.10)

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
                sx = int(round(0.5 * px + 0.5 * cx))
                sy = int(round(0.5 * py + 0.5 * cy))
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

    def classify_two_corners(self, corners_3d, plane_normal):
        """恰好 2 个可见角点时，判定二者是相邻还是对角。

        依据平面内间距：正方形相邻角点间距≈s，对角角点间距≈s√2（区分度~41%）。
        就近判定，边界在 s·(1+√2)/2 ≈ 1.207·s；间距明显超出合理范围时返回
        'unknown'（类别边长选错/角点误检），交由上层走兜底逻辑。

        返回: 'adjacent' | 'diagonal' | 'unknown'
        """
        if corners_3d is None or len(corners_3d) != 2 or plane_normal is None:
            return 'unknown'
        n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
        e = corners_3d[1] - corners_3d[0]
        e = e - np.dot(e, n_hat) * n_hat
        d = float(np.linalg.norm(e))
        s = EXPECTED_SIDE_M
        if d < 1e-4:
            return 'unknown'
        # 合理带：[0.75s, s√2 + 0.25s]，出界即兜底
        if d < 0.75 * s or d > s * math.sqrt(2.0) + 0.25 * s:
            return 'unknown'
        return 'diagonal' if abs(d - s * math.sqrt(2.0)) < abs(d - s) else 'adjacent'

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
                    err = abs(le - EXPECTED_SIDE_M)
                    if err < best_err:
                        best_err = err
                        best_pair = e

            if best_pair is not None:
                u = best_pair / (np.linalg.norm(best_pair) + 1e-8)

        # 恰好 2 个角点且判为对角：连线是对角线，绕法向量转 45° 回到边方向。
        # 正方形对角线平分两边夹角，±45° 给出两条互相垂直的边，取其一即可，
        # 后续 v = n×u 和"顺时针最小角"规范化会自动定死唯一朝向。
        if u is not None and self.classify_two_corners(corners_3d, plane_normal) == 'diagonal':
            e_diag = u - np.dot(u, n_hat) * n_hat
            le = np.linalg.norm(e_diag)
            if le > 1e-8:
                e_diag = e_diag / le
                w = np.cross(n_hat, e_diag)
                u = (e_diag + w) / math.sqrt(2.0)

        if u is None:
            u = self.prev_u_vec if self.prev_u_vec is not None else ref

        u = u - np.dot(u, n_hat) * n_hat
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(n_hat, u)
        v = v / (np.linalg.norm(v) + 1e-8)

        # ===== 新逻辑：基于相机X轴投影寻找顺时针夹角最小方向作为 U 轴 =====
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
            alpha = 0.40
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
            A, B = uv_pts[0], uv_pts[1]
            mid = 0.5 * (A + B)
            kind = self.classify_two_corners(corners_3d, plane_normal)
            if kind == 'diagonal':
                # 对角：正方形对角线互相平分，中心就是中点，无方向歧义
                center_uv = mid
            else:
                # 相邻（或 unknown 兜底）：中点沿垂线偏移半边长，方向用掩膜像素投票消歧
                t = B - A
                nt = np.linalg.norm(t)
                if nt < 1e-8:
                    center_uv = mid
                else:
                    t = t / nt
                    n2 = np.array([-t[1], t[0]], dtype=np.float64)
                    c1 = mid + 0.5 * EXPECTED_SIDE_M * n2
                    c2 = mid - 0.5 * EXPECTED_SIDE_M * n2

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
            h = 0.5 * EXPECTED_SIDE_M
            cands = [
                corner + np.array([ h,  h], dtype=np.float64),
                corner + np.array([ h, -h], dtype=np.float64),
                corner + np.array([-h,  h], dtype=np.float64),
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
    def draw_corners(image: np.ndarray, corners):
        out = image.copy()
        for x, y in corners:
            cv2.circle(out, (x, y), 5, (255, 255, 0), -1)
            cv2.circle(out, (x, y), 9, (0, 128, 255), 2)
        cv2.putText(out, f"visible right-angle corners: {len(corners)}", (10, 28),
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


def get_largest_mask_from_result(result, thresh=MASK_THRESH):
    if result.masks is None or len(result.masks) == 0:
        return None
    masks = result.masks.data.cpu().numpy()
    merged = (np.any(masks > thresh, axis=0).astype(np.uint8) * 255)
    largest = PlaneAndCornerEstimator.largest_component(merged)
    return PlaneAndCornerEstimator.fill_holes(largest)


def main(args=None):
    if not os.path.exists(MODEL_PATH):
        print(f"✗ model not found: {MODEL_PATH}")
        return

    # ===== ROS 2 节点初始化 =====
    rclpy.init(args=args)
    node = rclpy.create_node('surface_center_estimator')
    pose_pub = node.create_publisher(PoseStamped, '/surface_pose', 10)

    # 缓存状态，供服务回调和显示循环使用
    global_state = {
        'pose': None,
        'last_success_time': 0.0,
        'vis': None,
    }

    # 定义服务回调函数：收到请求时才进行一次推理
    def handle_get_pick_pose(request, response):
        node.get_logger().info(f'Service called for {request.object_name}: running inference...')
        color, depth = cam.read()
        if color is None:
            response.pick_pose = PoseStamped()
            response.success = False
            node.get_logger().warn('Service: failed to read camera frame.')
            return response

        capture_stamp = node.get_clock().now()
        result = model.predict(source=color, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]
        mask_u8 = get_largest_mask_from_result(result, MASK_THRESH)

        vis = color.copy()
        corners = []
        center_3d = None
        axis_u, axis_v, normal = None, None, None

        if mask_u8 is not None:
            pts3d = estimator.extract_mask_points_3d(mask_u8, depth)
            if pts3d is not None:
                plane = estimator.ransac_plane(pts3d)
                if plane is not None:
                    origin, normal = plane
                    corners = estimator.extract_visible_right_angle_vertices(mask_u8, origin, normal)
                    corners = estimator.temporal_stabilize_corners(corners)
                    corners_3d = estimator.corners_px_to_3d(corners, origin, normal)
                    axis_u, axis_v = estimator.estimate_uv_axes(corners_3d, normal)

                    x_axis_ref = np.array([1.0, 0.0, 0.0])
                    angle_u = np.degrees(np.arccos(np.clip(abs(np.dot(axis_u, x_axis_ref)), 0.0, 1.0)))
                    angle_v = np.degrees(np.arccos(np.clip(abs(np.dot(axis_v, x_axis_ref)), 0.0, 1.0)))
                    min_angle = min(angle_u, angle_v)

                    ys_m, xs_m = np.where(mask_u8 > 0)
                    cx_px = float(np.mean(xs_m))
                    cy_px = float(np.mean(ys_m))
                    centroid_3d = estimator.ray_plane_point(cx_px, cy_px, origin, normal)
                    mask_centroid_uv = None
                    if centroid_3d is not None:
                        dc = centroid_3d - origin
                        mask_centroid_uv = np.array([np.dot(dc, axis_u), np.dot(dc, axis_v)], dtype=np.float64)

                    center_3d, _ = estimator.compute_center_from_corner_count(
                        corners_3d, origin, axis_u, axis_v, mask_centroid_uv, mask_u8=mask_u8, plane_normal=normal)

                    if center_3d is not None:
                        n_hat = normal / (np.linalg.norm(normal) + 1e-8)
                        R = np.column_stack((axis_u, axis_v, n_hat))
                        q = rotmat2quat(R)

                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = capture_stamp.to_msg()
                        pose_msg.header.frame_id = SOURCE_FRAME
                        pose_msg.pose.position.x = float(center_3d[0])
                        pose_msg.pose.position.y = float(center_3d[1])
                        pose_msg.pose.position.z = float(center_3d[2])
                        pose_msg.pose.orientation.x = float(q[0])
                        pose_msg.pose.orientation.y = float(q[1])
                        pose_msg.pose.orientation.z = float(q[2])
                        pose_msg.pose.orientation.w = float(q[3])

                        transformed = False
                        last_tf_err = None
                        for query_time, tag in ((capture_stamp, "exact"), (Time(), "latest")):
                            try:
                                tf = tf_buffer.lookup_transform(
                                    TARGET_FRAME, SOURCE_FRAME, query_time,
                                    timeout=Duration(seconds=TF_TIMEOUT_SEC))
                                world_pose = do_transform_pose(pose_msg.pose, tf)
                                pose_msg.pose = world_pose
                                pose_msg.header.frame_id = TARGET_FRAME
                                transformed = True
                                break
                            except Exception as e:
                                last_tf_err = e
                        if not transformed:
                            node.get_logger().warn(
                                f"TF {TARGET_FRAME}<-{SOURCE_FRAME} unavailable ({last_tf_err}); keeping {SOURCE_FRAME} frame")

                        pose_pub.publish(pose_msg)
                        global_state['pose'] = pose_msg
                        global_state['last_success_time'] = time.monotonic()

                        vis = estimator.draw_mask_overlay(vis, mask_u8)
                        vis = estimator.draw_corners(vis, corners)
                        vis = estimator.draw_uv_axes_and_center(vis, center_3d, axis_u, axis_v, normal=normal, min_angle=min_angle)
                        global_state['vis'] = vis

                        response.pick_pose = pose_msg
                        response.success = True
                        node.get_logger().info(
                            f'Service: inference OK, corners={len(corners_3d)}, min_angle={min_angle:.1f}°')
                        return response

        response.pick_pose = PoseStamped()
        response.success = False
        node.get_logger().warn('Service: inference produced no valid pose.')
        return response

    # 创建服务端，服务名完全匹配现有 C++ 的请求 "get_pick_pos"
    srv = node.create_service(GetPickPos, 'get_pick_pos', handle_get_pick_pose)

    # TF：监听 arm control_node 广播的 world→Link_4(动态,100Hz) 和 Link_4→camera_link(静态)。
    # spin_thread=True：TF 用独立线程接收——否则单线程主循环里 YOLO(CPU,数百 ms)会饿死
    #   订阅回调，导致 buffer 最新 TF 恒定滞后约一个推理周期(≈1s)，按捕获时刻查必然
    #   "extrapolation into the future"。独立线程让 buffer 始终最新，capture_stamp 精确命中。
    # buffer 保存历史(默认 10s)，用捕获时刻查询得到"拍到物体那一帧"的相机姿态而非最新姿态，
    # 这样推理延迟期间相机移动也不会引入 world 坐标漂移。
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=True)

    cam = RealSenseCamera()
    model = YOLO(MODEL_PATH)
    estimator = PlaneAndCornerEstimator(cam.intr, cam.depth_scale)

    print("\nOn-demand inference mode: 推理仅在收到服务请求时触发. Press q to quit.")

    VIS_DISPLAY_SEC = 3.0  # 推理结果展示时长，之后恢复实时画面

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)

            if not os.environ.get('HEADLESS'):
                color, _ = cam.read()
                if color is not None:
                    age = time.monotonic() - global_state['last_success_time']
                    if global_state['vis'] is not None and age < VIS_DISPLAY_SEC:
                        vis = global_state['vis']
                    else:
                        vis = color
                    cv2.imshow("camera: Mask Visible Right-Angle Corners", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.02)

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
