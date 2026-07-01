#!/usr/bin/env bash
# run_neweyes.sh — neweyes 视觉节点一键启动
# 用法：
#   bash run_neweyes.sh          # 正常启动
#   bash run_neweyes.sh --build  # 先编译再启动
#   bash run_neweyes.sh -h       # 查看帮助
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${WS_DIR:-$SCRIPT_DIR}"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$WS_DIR/install/setup.bash"
AUTO_BUILD="${AUTO_BUILD:-false}"

# ===== 帮助 =====
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "用法: bash run_neweyes.sh [--build] [--headless]"
  echo "  --build      启动前先执行 colcon build"
  echo "  --headless   不弹可视化窗口，Ctrl+C 退出"
  echo "  环境变量 WS_DIR 可覆盖工作空间路径（默认脚本所在目录）"
  exit 0
fi

HEADLESS_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--build" ]] && AUTO_BUILD=true
  [[ "$arg" == "--headless" ]] && HEADLESS_MODE=true
done

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
if [[ "$HEADLESS_MODE" == "true" ]]; then
  echo "[neweyes] 无头模式，按 Ctrl+C 退出"
else
  echo "[neweyes] 按 q 退出可视化窗口"
fi
echo ""

if [[ "$HEADLESS_MODE" == "true" ]]; then
  HEADLESS=1 ros2 run my_pick_pipeline pose_estimator
else
  ros2 run my_pick_pipeline pose_estimator
fi
