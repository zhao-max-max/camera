#!/usr/bin/env bash
# run_neweyes.sh — neweyes 视觉节点一键启动
# 用法：
#   bash run_neweyes.sh --1            # 使用相机 213622078104
#   bash run_neweyes.sh --2            # 使用相机 043322075459
#   bash run_neweyes.sh --1 --build    # 先编译再启动
#   bash run_neweyes.sh --1 --required # 按需推理，收到服务请求才推理
#   bash run_neweyes.sh -h             # 查看帮助
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${WS_DIR:-$SCRIPT_DIR}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$WS_DIR/install/setup.bash"
AUTO_BUILD="${AUTO_BUILD:-false}"
VENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
ENTRY_POINT="$WS_DIR/install/my_pick_pipeline/lib/my_pick_pipeline/pose_estimator"

SERIAL_1="213622078104"
SERIAL_2="043322075459"

# ===== 选择 Python =====
if python3 -c "import pyrealsense2, ultralytics, cv2" 2>/dev/null; then
  PYTHON_BIN="python3"
  echo "[neweyes] 使用系统 Python: $(which python3)"
elif [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
  echo "[neweyes] 使用虚拟环境 Python: $VENV_PYTHON"
else
  echo "[neweyes] ERROR: 找不到可用的 Python 环境（系统缺少依赖，虚拟环境也不存在 $VENV_DIR）" >&2
  exit 1
fi

# ===== 帮助 =====
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "用法: bash run_neweyes.sh --1|--2 [--build] [--headless] [--required]"
  echo "  --1        使用相机 ${SERIAL_1}"
  echo "  --2        使用相机 ${SERIAL_2}"
  echo "  --build    启动前先执行 colcon build"
  echo "  --headless 不弹可视化窗口，Ctrl+C 退出"
  echo "  --required 按需推理；不加则默认持续实时推理"
  echo "  环境变量 WS_DIR 可覆盖工作空间路径（默认脚本所在目录）"
  exit 0
fi

HEADLESS_MODE=false
REQUIRED_MODE=false
CAMERA_SERIAL=""
for arg in "$@"; do
  [[ "$arg" == "--build" ]] && AUTO_BUILD=true
  [[ "$arg" == "--headless" ]] && HEADLESS_MODE=true
  [[ "$arg" == "--required" ]] && REQUIRED_MODE=true
  [[ "$arg" == "--1" ]] && CAMERA_SERIAL="$SERIAL_1"
  [[ "$arg" == "--2" ]] && CAMERA_SERIAL="$SERIAL_2"
done

if [[ -z "$CAMERA_SERIAL" ]]; then
  echo "[neweyes] ERROR: 请指定相机编号：--1 (${SERIAL_1}) 或 --2 (${SERIAL_2})" >&2
  exit 1
fi
export CAMERA_SERIAL

source_setup() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

# ===== 检查 ROS 2 =====
[[ -f "$ROS_SETUP" ]] || { echo "[neweyes] ERROR: 找不到 ROS 2 Humble ($ROS_SETUP)" >&2; exit 1; }
source_setup "$ROS_SETUP"

# ===== 可选编译 =====
if [[ "$AUTO_BUILD" == "true" ]]; then
  echo "[neweyes] 编译中..."
  (cd "$WS_DIR" && colcon build --symlink-install 2>&1)
  echo "[neweyes] 编译完成"
fi

# ===== source 工作空间 =====
[[ -f "$WS_SETUP" ]] \
  || { echo "[neweyes] ERROR: install/setup.bash 不存在，请先编译（--build 或 colcon build）" >&2; exit 1; }
source_setup "$WS_SETUP"

echo "[neweyes] 启动视觉节点 pose_estimator ..."
echo "[neweyes] 服务名: get_pick_pos  帧: camera_link"
echo "[neweyes] 相机序列号: $CAMERA_SERIAL"
echo "[neweyes] Python: $PYTHON_BIN"
if [[ "$HEADLESS_MODE" == "true" ]]; then
  echo "[neweyes] 无头模式，按 Ctrl+C 退出"
else
  echo "[neweyes] 按 q 退出可视化窗口"
fi
echo ""

# 把工作空间的 robot_msgs 加到 PYTHONPATH，让 venv Python 能找到本地编译的消息类型
ROBOT_MSGS_PATH="$WS_DIR/install/robot_msgs/local/lib/python3.10/dist-packages"
export PYTHONPATH="$ROBOT_MSGS_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="$WS_DIR/best6.27.pt"

PY_ARGS=()
[[ "$REQUIRED_MODE" == "true" ]] && PY_ARGS+=(--required)

if [[ "$HEADLESS_MODE" == "true" ]]; then
  HEADLESS=1 "$PYTHON_BIN" "$ENTRY_POINT" "${PY_ARGS[@]}"
else
  "$PYTHON_BIN" "$ENTRY_POINT" "${PY_ARGS[@]}"
fi
