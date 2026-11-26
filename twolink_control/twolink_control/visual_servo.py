#!/usr/bin/env python3
"""ArUco 마커 추종 노드 (역기구학 기반 Position-Based Visual Servoing).

카메라는 twolink 로봇 끝(link2)에 달려 수평 방향을 바라본다.
두 관절(q1, q2)은 모두 수직축(+Z) 회전 -> 수평면 2자유도.

[1] 순기구학 + 카메라 측정 -> 마커의 base 좌표
      o_c(q) = L1*[c1, s1] + R(q1+q2) [cx, cy]         (카메라 광학중심)
      d(q)   = [cos(q1+q2+yaw), sin(q1+q2+yaw)]          (광축 방향)
      x(q)   = [sin(..), -cos(..)]                         (영상 오른쪽 방향)
      p      = o_c + X*x + Z*d                             (X, Z: 카메라 좌표계 마커 위치)

[2] 목표: 마커가 광축 위(영상 좌우 중앙), 거리 D 에 오도록
      p = L1*[c1, s1] + R(q1+q2) [cx + D, cy]
    -> v = [cx, cy] + D*[cos(yaw), sin(yaw)] (joint2 -> 마커, link2 좌표계) 를
       길이 r2 = |v|, 각도 오프셋 b = atan2(vy, vx) 인 가상 link2 로 보면
       일반 2링크 역기구학과 같다.
      cos(g) = (|p|^2 - L1^2 - r2^2) / (2 L1 r2)          (g: 가상 팔꿈치 각)
      q1     = atan2(py, px) - atan2(r2 sin g, L1 + r2 cos g)
      q2     = g - b
    두 해(g = +-acos) 중 현재 자세에 가까운 해를 선택.
    팔이 닿지 않는 거리면 r2 를 가능한 범위로 줄이거나 늘려 중앙 정렬을 우선한다.

[3] 목표각으로 속도 제한을 두고 부드럽게 이동 -> des_value 발행

입력  /aruco/pose (geometry_msgs/Pose), robot_state (twolink_msgs/XYpos)
출력  des_value (twolink_msgs/CMD), /visual_servo/markers (RViz 표시)
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from twolink_msgs.msg import CMD, XYpos


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def camera_fk(q1, q2, L1, cx, cy, yaw):
    """카메라 광학중심 o, 광축 방향 d, 영상 오른쪽 방향 x (모두 base 수평면)."""
    t12 = q1 + q2
    o = np.array([L1 * math.cos(q1) + cx * math.cos(t12) - cy * math.sin(t12),
                  L1 * math.sin(q1) + cx * math.sin(t12) + cy * math.cos(t12)])
    th = t12 + yaw
    d = np.array([math.cos(th), math.sin(th)])
    x = np.array([math.sin(th), -math.cos(th)])
    return o, d, x


def camera_ik(p, D, L1, cx, cy, yaw, q_ref):
    """마커 위치 p 를 카메라 정면 거리 D 에 두는 관절각 (q_ref 에 가까운 해).

    반환: (q, D_actual, reachable)
    """
    rho = float(np.hypot(p[0], p[1]))
    c = np.array([cx, cy])                       # link2 좌표계 카메라 위치
    u = np.array([math.cos(yaw), math.sin(yaw)])  # link2 좌표계 광축 방향
    v = c + D * u                                 # 가상 link2 벡터 (joint2 -> 마커)
    r2 = float(np.hypot(v[0], v[1]))
    reachable = abs(L1 - r2) <= rho <= L1 + r2
    if not reachable:
        # 거리를 맞출 수 없으면 가능한 가장 가까운 r2 로 (중앙 정렬은 유지)
        r2 = min(max(r2, abs(rho - L1) + 1e-6), rho + L1 - 1e-6)
        cu = float(c @ u)
        D = -cu + math.sqrt(max(cu * cu - float(c @ c) + r2 * r2, 0.0))
        v = c + D * u
        r2 = float(np.hypot(v[0], v[1]))
    b = math.atan2(v[1], v[0])
    cg = (rho * rho - L1 * L1 - r2 * r2) / (2.0 * L1 * r2)
    g0 = math.acos(max(-1.0, min(1.0, cg)))
    best = None
    for g in (g0, -g0):
        q1 = math.atan2(p[1], p[0]) - math.atan2(r2 * math.sin(g), L1 + r2 * math.cos(g))
        q2 = g - b
        # 현재 자세와 가장 가까운 동치각으로 (2pi 점프 방지)
        q = np.array([q_ref[0] + wrap(q1 - q_ref[0]), q_ref[1] + wrap(q2 - q_ref[1])])
        cost = float(np.sum((q - q_ref) ** 2))
        if best is None or cost < best[1]:
            best = (q, cost)
    return best[0], D, reachable


class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo')

        p = self.declare_parameter
        self.desired_distance = p('desired_distance', 0.30).value   # [m]
        self.max_joint_vel = p('max_joint_vel', 0.6).value  # [rad/s]
        self.gain = p('gain', 3.0).value                    # 목표각 추종 게인 [1/s]
        self.filter_alpha = p('filter_alpha', 0.3).value    # 마커 base 좌표 저역통과 (1: 필터 없음)
        self.deadband = p('deadband', 0.01).value           # 목표각 오차 무시 범위 [rad]
        self.marker_timeout = p('marker_timeout', 0.3).value  # [s]
        self.state_timeout = p('state_timeout', 0.5).value    # [s]
        self.rate = p('rate', 20.0).value                   # [Hz]
        self.q1_lim = (p('q1_min', -1.57).value, p('q1_max', 1.57).value)
        self.q2_lim = (p('q2_min', -2.4).value, p('q2_max', 2.4).value)
        # 로봇 / 카메라 장착 기하 (link2 좌표계 기준)
        self.L1 = p('link1_length', 0.143).value
        self.cam_x = p('camera_offset_x', 0.06).value
        self.cam_y = p('camera_offset_y', 0.05).value
        self.cam_yaw = p('camera_yaw', 0.0).value
        self.cam_z = p('camera_height', 0.20).value   # RViz 표시용 카메라 높이 [m]
        self.dry_run = p('dry_run', False).value

        self.q = None
        self.q_time = 0.0
        self.p_marker = None        # 필터된 마커 base 좌표 [x, y]
        self.marker_y = 0.0         # 카메라 좌표계 Y (표시용 높이)
        self.marker_time = 0.0
        self.q_cmd = None
        self.q_goal = None
        self.tracking = False
        self.commanded = False
        self.last_log = 0.0

        self.create_subscription(Pose, '/aruco/pose', self.marker_callback, 10)
        self.create_subscription(XYpos, 'robot_state', self.state_callback, 10)
        self.cmd_pub = self.create_publisher(CMD, 'des_value', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/visual_servo/markers', 10)
        self.dt = 1.0 / self.rate
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f'Visual servo (IK) start: distance={self.desired_distance:.2f} m'
            + (' [DRY RUN - no commands sent]' if self.dry_run else ''))

    # ---------------- 콜백 ----------------
    def state_callback(self, msg):
        self.q = np.array([msg.qpos1, msg.qpos2], dtype=float)
        self.q_time = time.monotonic()

    def marker_callback(self, msg):
        X, Z = msg.position.x, msg.position.z
        if self.q is None or not (math.isfinite(X) and math.isfinite(Z)) or Z <= 0.0:
            return
        # [1] 순기구학으로 마커의 base 좌표 계산
        o, d, x = camera_fk(self.q[0], self.q[1], self.L1, self.cam_x, self.cam_y, self.cam_yaw)
        p_new = o + X * x + Z * d
        now = time.monotonic()
        if self.p_marker is None or now - self.marker_time > self.marker_timeout:
            self.p_marker = p_new
        else:
            a = self.filter_alpha
            self.p_marker = a * p_new + (1.0 - a) * self.p_marker
        self.marker_y = msg.position.y
        self.marker_time = now

    # ---------------- 제어 루프 ----------------
    def control_loop(self):
        now = time.monotonic()
        if self.q is None or now - self.q_time > self.state_timeout:
            self.log_throttled('Waiting for robot_state... (serial_node / port 확인)')
            return
        if self.q_cmd is None:
            self.q_cmd = self.q.copy()

        marker_ok = self.p_marker is not None and now - self.marker_time < self.marker_timeout
        if not marker_ok:
            if self.tracking:
                self.get_logger().warning('Marker lost -> hold position')
                self.q_cmd = self.q.copy()
                self.tracking = False
            if self.commanded:
                self.publish(self.q_cmd)
            self.log_throttled('Waiting for marker...')
            return
        if not self.tracking:
            self.get_logger().info('Marker found -> tracking')
            self.q_cmd = self.q.copy()
            self.tracking = True

        # [2] 역기구학으로 목표 관절각
        q_goal, D, reachable = camera_ik(
            self.p_marker, self.desired_distance, self.L1,
            self.cam_x, self.cam_y, self.cam_yaw, self.q_cmd)
        q_goal[0] = np.clip(q_goal[0], *self.q1_lim)
        q_goal[1] = np.clip(q_goal[1], *self.q2_lim)
        self.q_goal = q_goal

        # [3] 속도 제한을 두고 목표각으로 이동
        err = q_goal - self.q_cmd
        err[np.abs(err) < self.deadband] = 0.0
        q_dot = self.gain * err
        peak = np.max(np.abs(q_dot))
        if peak > self.max_joint_vel:
            q_dot *= self.max_joint_vel / peak
        self.q_cmd = self.q_cmd + q_dot * self.dt

        self.publish(self.q_cmd)
        self.commanded = True
        self.publish_viz(D)
        self.log_throttled(
            f'marker base=({self.p_marker[0]:+.3f},{self.p_marker[1]:+.3f}) '
            f'| IK goal q=({q_goal[0]:+.3f},{q_goal[1]:+.3f})'
            f'{"" if reachable else f" [out of reach, D={D:.2f}]"} '
            f'| q=({self.q[0]:+.3f},{self.q[1]:+.3f}) cmd=({self.q_cmd[0]:+.3f},{self.q_cmd[1]:+.3f})')

    def publish(self, q):
        if self.dry_run:
            return
        msg = CMD()
        msg.xdes = 0.0
        msg.ydes = 0.0
        msg.q1des = float(q[0])
        msg.q2des = float(q[1])
        self.cmd_pub.publish(msg)

    def publish_viz(self, D):
        """RViz: 마커 위치(파랑), IK 목표 카메라 위치(초록), 목표 시선(노랑)."""
        stamp = self.get_clock().now().to_msg()
        z_marker = self.cam_z - self.marker_y
        o_goal, d_goal, _ = camera_fk(self.q_goal[0], self.q_goal[1],
                                      self.L1, self.cam_x, self.cam_y, self.cam_yaw)

        def mk(mid, mtype, rgb, scale):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = stamp
            m.ns = 'visual_servo'
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            m.pose.orientation.w = 1.0
            m.lifetime.nanosec = int(0.5e9)
            return m

        aruco = mk(0, Marker.CUBE, (0.1, 0.3, 1.0), (0.06, 0.06, 0.06))
        aruco.pose.position = Point(x=float(self.p_marker[0]), y=float(self.p_marker[1]), z=z_marker)
        goal = mk(1, Marker.SPHERE, (0.1, 0.9, 0.1), (0.03, 0.03, 0.03))
        goal.pose.position = Point(x=float(o_goal[0]), y=float(o_goal[1]), z=self.cam_z)
        ray = mk(2, Marker.LINE_STRIP, (1.0, 0.9, 0.1), (0.005, 0.0, 0.0))
        ray.points = [goal.pose.position,
                      Point(x=float(o_goal[0] + D * d_goal[0]),
                            y=float(o_goal[1] + D * d_goal[1]), z=self.cam_z)]
        self.viz_pub.publish(MarkerArray(markers=[aruco, goal, ray]))

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
