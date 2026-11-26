#!/usr/bin/env python3
"""ArUco 마커 시각 서보(visual servoing) 노드.

카메라가 twolink 로봇 끝(link2)에 달려 있고 수평 방향을 바라본다.
두 관절(q1, q2)은 모두 수직축(+Z) 회전이므로 수평면 안에서 2자유도를 가진다.
이 2자유도로 다음 두 가지를 동시에 제어한다.

  1) 좌우 중앙 정렬 : 카메라 좌표계 마커 X  -> 0
  2) 거리 유지      : 카메라 좌표계 마커 Z  -> desired_distance

입력
  /aruco/pose   (geometry_msgs/Pose)   카메라 좌표계 마커 위치 (OpenCV: x 오른쪽, y 아래, z 정면)
  robot_state   (twolink_msgs/XYpos)   현재 관절각 qpos1, qpos2 [rad]
출력
  des_value     (twolink_msgs/CMD)     목표 관절각 q1des, q2des [rad]

제어 방법 (resolved-rate / 자코비안 기반 위치 서보)
  e = [X, Z - d]
  e_dot = J(q) q_dot    (J: 관절 속도 -> 카메라에서 본 마커 위치 변화)
  q_dot = -K * J^+ e    (감쇠 최소자승 의사역행렬)
  q_cmd <- q_cmd + q_dot * dt    (속도 제한, 현재각 대비 최대 선행량 제한)
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from twolink_msgs.msg import CMD, XYpos


def camera_jacobian(q1, q2, X, Z, L1, cx, cy, yaw):
    """관절 속도 [dq1, dq2] -> 카메라 좌표계 마커 위치 변화 [dX, dZ] 자코비안.

    카메라 광학중심은 link2 좌표계의 (cx, cy), 광축은 link2 x축에서 yaw만큼 회전.
    마커는 world에 고정되어 있다고 가정한다.
      dX = -o_dot . x_hat + theta_dot * Z
      dZ = -o_dot . d_hat - theta_dot * X
    (d_hat: 광축 방향, x_hat: 영상 오른쪽 방향, theta = q1 + q2 + yaw)
    """
    s1, c1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    th = q1 + q2 + yaw
    d_hat = np.array([math.cos(th), math.sin(th)])
    x_hat = np.array([math.sin(th), -math.cos(th)])

    # 카메라 원점 위치 o(q)의 편미분
    do_dq2 = np.array([-cx * s12 - cy * c12, cx * c12 - cy * s12])
    do_dq1 = np.array([-L1 * s1, L1 * c1]) + do_dq2

    J = np.zeros((2, 2))
    for i, do in enumerate((do_dq1, do_dq2)):
        J[0, i] = -do @ x_hat + Z      # dX/dqi  (dtheta/dqi = 1)
        J[1, i] = -do @ d_hat - X      # dZ/dqi
    return J


class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo')

        p = self.declare_parameter
        # 목표
        self.desired_distance = p('desired_distance', 0.30).value   # [m]
        # 제어 게인 / 제한
        self.k_lateral = p('gain_lateral', 1.5).value     # [1/s]
        self.k_depth = p('gain_depth', 1.0).value         # [1/s]
        self.max_joint_vel = p('max_joint_vel', 0.6).value  # [rad/s]
        self.max_lead = p('max_lead', 0.25).value         # 목표각이 현재각보다 앞설 수 있는 최대값 [rad]
        self.damping = p('damping', 0.03).value           # 의사역행렬 감쇠 (특이점 대비)
        self.deadband_x = p('deadband_x', 0.005).value    # [m]
        self.deadband_z = p('deadband_z', 0.01).value     # [m]
        self.filter_alpha = p('filter_alpha', 0.5).value  # 마커 위치 저역통과 (1: 필터 없음)
        self.marker_timeout = p('marker_timeout', 0.3).value  # [s]
        self.state_timeout = p('state_timeout', 0.5).value    # [s]
        self.rate = p('rate', 20.0).value                 # [Hz]
        # 관절 제한 [rad]
        self.q1_lim = (p('q1_min', -1.57).value, p('q1_max', 1.57).value)
        self.q2_lim = (p('q2_min', -2.4).value, p('q2_max', 2.4).value)
        # 로봇 / 카메라 장착 기하 (link2 좌표계 기준)
        self.L1 = p('link1_length', 0.143).value
        self.cam_x = p('camera_offset_x', 0.06).value
        self.cam_y = p('camera_offset_y', 0.05).value
        self.cam_yaw = p('camera_yaw', 0.0).value
        # true면 목표각을 계산/출력만 하고 로봇에는 보내지 않음 (방향 확인용)
        self.dry_run = p('dry_run', False).value

        self.q = None
        self.q_time = 0.0
        self.marker = None          # 필터된 [X, Z]
        self.marker_time = 0.0
        self.q_cmd = None
        self.tracking = False
        self.commanded = False
        self.last_log = 0.0

        self.create_subscription(Pose, '/aruco/pose', self.marker_callback, 10)
        self.create_subscription(XYpos, 'robot_state', self.state_callback, 10)
        self.cmd_pub = self.create_publisher(CMD, 'des_value', 10)
        self.dt = 1.0 / self.rate
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'Visual servo start: distance={self.desired_distance:.2f} m'
            + (' [DRY RUN - no commands sent]' if self.dry_run else ''))

    # ---------------- 콜백 ----------------
    def state_callback(self, msg):
        self.q = np.array([msg.qpos1, msg.qpos2], dtype=float)
        self.q_time = time.monotonic()

    def marker_callback(self, msg):
        meas = np.array([msg.position.x, msg.position.z], dtype=float)
        if not np.all(np.isfinite(meas)) or meas[1] <= 0.0:
            return
        now = time.monotonic()
        if self.marker is None or now - self.marker_time > self.marker_timeout:
            self.marker = meas      # 새로 보이기 시작하면 필터 초기화
        else:
            a = self.filter_alpha
            self.marker = a * meas + (1.0 - a) * self.marker
        self.marker_time = now

    # ---------------- 제어 루프 ----------------
    def control_loop(self):
        now = time.monotonic()
        if self.q is None or now - self.q_time > self.state_timeout:
            self.log_throttled('Waiting for robot_state...')
            return
        if self.q_cmd is None:
            self.q_cmd = self.q.copy()

        marker_ok = self.marker is not None and now - self.marker_time < self.marker_timeout
        if not marker_ok:
            # 마커를 놓치면 그 자리에 정지 (현재 각도를 목표로 유지)
            if self.tracking:
                self.get_logger().warning('Marker lost -> hold position')
                self.q_cmd = self.q.copy()
                self.tracking = False
            if self.commanded:
                # 한 번이라도 추종을 시작했으면 마지막 위치를 계속 유지
                self.publish(self.q_cmd)
            self.log_throttled('Waiting for marker...')
            return
        if not self.tracking:
            self.get_logger().info('Marker found -> tracking')
            self.q_cmd = self.q.copy()
            self.tracking = True

        X, Z = self.marker
        e = np.array([X, Z - self.desired_distance])
        if abs(e[0]) < self.deadband_x:
            e[0] = 0.0
        if abs(e[1]) < self.deadband_z:
            e[1] = 0.0

        J = camera_jacobian(self.q_cmd[0], self.q_cmd[1], X, Z,
                            self.L1, self.cam_x, self.cam_y, self.cam_yaw)
        # 감쇠 최소자승 의사역행렬: J^T (J J^T + lambda^2 I)^-1
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (self.damping ** 2) * np.eye(2))
        K = np.diag([self.k_lateral, self.k_depth])
        q_dot = -J_pinv @ (K @ e)

        # 관절 한계에 걸려 더 못 가는 관절이 있으면, 남은 관절 하나로 좌우 중앙 정렬을 우선한다.
        lims = (self.q1_lim, self.q2_lim)
        blocked = [
            (self.q_cmd[i] <= lims[i][0] + 1e-3 and q_dot[i] < 0.0)
            or (self.q_cmd[i] >= lims[i][1] - 1e-3 and q_dot[i] > 0.0)
            for i in range(2)]
        if any(blocked) and not all(blocked):
            free = blocked.index(False)
            q_dot = np.zeros(2)
            if abs(J[0, free]) > 1e-3:
                q_dot[free] = -self.k_lateral * e[0] / J[0, free]
        elif all(blocked):
            q_dot = np.zeros(2)

        # 관절 속도 제한 (방향 유지한 채 스케일)
        peak = np.max(np.abs(q_dot))
        if peak > self.max_joint_vel:
            q_dot *= self.max_joint_vel / peak

        q_new = self.q_cmd + q_dot * self.dt
        # 실제 관절각보다 너무 앞서가지 않도록 제한 (급격한 움직임 방지)
        q_new = np.clip(q_new, self.q - self.max_lead, self.q + self.max_lead)
        q_new[0] = np.clip(q_new[0], *self.q1_lim)
        q_new[1] = np.clip(q_new[1], *self.q2_lim)
        self.q_cmd = q_new

        self.publish(self.q_cmd)
        self.commanded = True
        self.log_throttled(
            f'marker X={X:+.3f} Z={Z:.3f} m | err x={e[0]:+.3f} z={e[1]:+.3f} | '
            f'q=({self.q[0]:+.3f},{self.q[1]:+.3f}) -> cmd=({self.q_cmd[0]:+.3f},{self.q_cmd[1]:+.3f})')

    def publish(self, q):
        if self.dry_run:
            return
        msg = CMD()
        msg.xdes = 0.0
        msg.ydes = 0.0
        msg.q1des = float(q[0])
        msg.q2des = float(q[1])
        self.cmd_pub.publish(msg)

    def log_throttled(self, text, period=0.5):
        now = time.monotonic()
        if now - self.last_log >= period:
            self.get_logger().info(text)
            self.last_log = now


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
