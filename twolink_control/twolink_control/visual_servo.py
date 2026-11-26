#!/usr/bin/env python3
"""ArUco 마커 추종 노드 (역기구학 기반 Position-Based Visual Servoing).

카메라는 twolink 로봇 끝(link2)에 달려 수평 방향을 바라본다.
두 관절(q1, q2)은 모두 수직축(+Z) 회전 -> 수평면 2자유도.

[1] 순기구학 + 카메라 측정 -> 마커의 base 좌표
      o_c(q) = L1*[c1, s1] + R(q1+q2) [cx, cy]         (카메라 광학중심)
      d(q)   = [cos(q1+q2+yaw), sin(q1+q2+yaw)]          (광축 방향)
      x(q)   = [sin(..), -cos(..)]                         (영상 오른쪽 방향)
      p      = o_c + X*x + Z*d                             (X: 영상 중앙 세로선과 마커의 좌우 거리, Z: 마커 거리)

[2] 목표: 마커가 광축 위(영상 좌우 중앙), 거리 D 에 오도록
      p = L1*[c1, s1] + R(q1+q2) [cx + D, cy]
    -> v = [cx, cy] + D*[cos(yaw), sin(yaw)] (joint2 -> 마커, link2 좌표계) 를
       길이 r2 = |v|, 각도 오프셋 b = atan2(vy, vx) 인 가상 link2 로 보면
       일반 2링크 역기구학과 같다.
      cos(g) = (|p|^2 - L1^2 - r2^2) / (2 L1 r2)          (g: 가상 팔꿈치 각)
      q1     = atan2(py, px) - atan2(r2 sin g, L1 + r2 cos g)
      q2     = g - b
    두 해(g = +-acos) 중 현재 자세에 가까운 해를 선택.
    D 로는 팔이 닿지 않거나 관절 한계를 넘으면, 한계 안에서 마커를 중앙에 두는
    해 중 거리가 D 에 가장 가까운 해를 고른다 (중앙 정렬 우선).

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
from std_msgs.msg import Bool
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
    sols, D, reachable = camera_ik_all(p, D, L1, cx, cy, yaw, q_ref)
    best = min(sols, key=lambda q: float(np.sum((q - q_ref) ** 2)))
    return best, D, reachable


def camera_ik_all(p, D, L1, cx, cy, yaw, q_ref):
    """camera_ik 의 두 해(팔꿈치 +/-)를 모두 반환: ([q_a, q_b], D_actual, reachable)"""
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
    sols = []
    for g in (g0, -g0):
        q1 = math.atan2(p[1], p[0]) - math.atan2(r2 * math.sin(g), L1 + r2 * math.cos(g))
        q2 = g - b
        # 현재 자세와 가장 가까운 동치각으로 (2pi 점프 방지)
        sols.append(np.array([q_ref[0] + wrap(q1 - q_ref[0]), q_ref[1] + wrap(q2 - q_ref[1])]))
    return sols, D, reachable


def camera_ik_limited(p, D, L1, cx, cy, yaw, q_ref, q1_lim, q2_lim,
                      d_min=0.12, d_max=0.80, d_step=0.01):
    """관절 한계를 고려한 역기구학.

    목표 거리 D 의 해가 관절 한계 밖이면, 마커를 영상 중앙에 두는 해 중에서
    한계 안에 있고 거리가 D 에 가장 가까운 해를 고른다 (중앙 정렬 우선).
    반환: (q, D_actual, ok)  ok=False 면 한계 안의 해가 없어 가장 가까운 해를 잘라서 반환
    """
    def inside(q):
        return q1_lim[0] <= q[0] <= q1_lim[1] and q2_lim[0] <= q[1] <= q2_lim[1]

    n = int(round((d_max - d_min) / d_step))
    cands = sorted({D} | {d_min + i * d_step for i in range(n + 1)}, key=lambda d: abs(d - D))
    for d in cands:
        sols, d_act, reachable = camera_ik_all(p, d, L1, cx, cy, yaw, q_ref)
        if not reachable:
            continue
        ok = [q for q in sols if inside(q)]
        if ok:
            return min(ok, key=lambda q: float(np.sum((q - q_ref) ** 2))), d_act, True
    q, d_act, _ = camera_ik(p, D, L1, cx, cy, yaw, q_ref)
    return np.array([np.clip(q[0], *q1_lim), np.clip(q[1], *q2_lim)]), d_act, False


class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo')

        p = self.declare_parameter
        self.desired_distance = p('desired_distance', 0.30).value   # [m]
        self.max_joint_vel = p('max_joint_vel', 2.0).value  # [rad/s]
        self.gain = p('gain', 6.0).value                    # 목표각 추종 게인 [1/s]
        self.filter_alpha = p('filter_alpha', 0.5).value    # 마커 base 좌표 저역통과 (1: 필터 없음)
        self.deadband = p('deadband', 0.008).value          # 목표각 오차 무시 범위 [rad]
        # 마커 인식 실패 프레임이 lost_frames 개 연속이면 즉시 정지 (1: 한 프레임만 놓쳐도 정지)
        self.lost_frames = p('lost_frames', 1).value
        # /aruco/detected 자체가 끊긴 경우(인식 노드 정지 등)의 안전 타임아웃.
        # 화면에 마커가 없을 때는 detected=False 로 즉시 정지하므로, 이 값은 인식 노드가
        # 잠깐 밀릴 때 불필요하게 멈추지 않도록 여유 있게 둔다.
        self.marker_timeout = p('marker_timeout', 0.5).value  # [s]
        # 목표각이 실제 관절각보다 앞설 수 있는 최대값 [rad].
        # 작을수록 정지 명령 후 로봇이 덜 미끄러지고, 클수록 빠르게 움직인다.
        self.max_lead = p('max_lead', 0.4).value
        self.state_timeout = p('state_timeout', 0.5).value    # [s]
        self.rate = p('rate', 20.0).value                   # [Hz]
        # 관절 회전 제한 (기본: 사용 안 함). use_joint_limits:=true 면 아래 범위 적용
        self.use_joint_limits = p('use_joint_limits', False).value
        if self.use_joint_limits:
            self.q1_lim = (p('q1_min', -1.57).value, p('q1_max', 1.57).value)
            self.q2_lim = (p('q2_min', -2.4).value, p('q2_max', 2.4).value)
        else:
            self.q1_lim = (-math.inf, math.inf)
            self.q2_lim = (-math.inf, math.inf)
        # 로봇 / 카메라 장착 기하 (link2 좌표계 기준)
        self.L1 = p('link1_length', 0.143).value
        self.cam_x = p('camera_offset_x', 0.06).value
        self.cam_y = p('camera_offset_y', 0.05).value
        self.cam_yaw = p('camera_yaw', 0.0).value
        self.cam_z = p('camera_height', 0.20).value   # RViz 표시용 카메라 높이 [m]
        self.dry_run = p('dry_run', False).value

        self.q = None
        self.qvel = np.zeros(2)
        self.q_time = 0.0
        self.missed = 0             # 연속 인식 실패 프레임 수
        self.p_marker = None        # 필터된 마커 base 좌표 [x, y]
        self.marker_y = 0.0         # 카메라 좌표계 Y (표시용 높이)
        self.marker_time = 0.0
        self.q_cmd = None
        self.q_goal = None
        self.tracking = False
        self.commanded = False
        self.last_log = 0.0

        self.create_subscription(Pose, '/aruco/pose', self.marker_callback, 10)
        self.create_subscription(Bool, '/aruco/detected', self.detected_callback, 10)
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
        self.qvel = np.array([msg.qvel1, msg.qvel2], dtype=float)
        if not np.all(np.isfinite(self.qvel)):
            self.qvel = np.zeros(2)
        self.q_time = time.monotonic()

    def estimate_q(self):
        """지금 이 순간의 실제 관절각 추정 (마지막 측정 + 속도 x 경과시간)."""
        age = min(time.monotonic() - self.q_time, 0.15)
        return self.q + self.qvel * age

    def detected_callback(self, msg):
        if msg.data:
            self.missed = 0
            return
        self.missed += 1
        if self.missed >= self.lost_frames and self.tracking:
            self.stop_now('Marker lost')

    def stop_now(self, reason):
        """추종을 멈추고 로봇을 지금 위치에 즉시 정지시킨다."""
        self.tracking = False
        if self.q is None:
            return
        self.q_cmd = self.estimate_q()
        self.publish(self.q_cmd)    # 타이머를 기다리지 않고 바로 정지 명령
        self.get_logger().warning(f'{reason} -> STOP at q=({self.q_cmd[0]:+.3f},{self.q_cmd[1]:+.3f})')

    def marker_callback(self, msg):
        X, Z = msg.position.x, msg.position.z
        if self.q is None or not (math.isfinite(X) and math.isfinite(Z)) or Z <= 0.0:
            return
        # [1] 순기구학으로 마커의 base 좌표 계산
        o, d, x = camera_fk(self.q[0], self.q[1], self.L1, self.cam_x, self.cam_y, self.cam_yaw)
        p_new = o + X * x + Z * d
        now = time.monotonic()
        if self.p_marker is None or not self.tracking or now - self.marker_time > self.marker_timeout:
            self.p_marker = p_new       # 다시 보이기 시작하면 필터 초기화 (이전 위치 무시)
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

        marker_ok = (self.p_marker is not None
                     and now - self.marker_time < self.marker_timeout
                     and self.missed < self.lost_frames)
        if not marker_ok:
            if self.tracking:
                self.stop_now('Marker timeout')
            # 정지 명령은 stop_now() 에서 한 번만 보낸다. serial_node 가 마지막 목표를
            # 계속 반복 전송하므로 여기서 다시 보내지 않는다 (보드 리셋 후 예전 자세로
            # 끌려가지 않게).
            self.log_throttled('Waiting for marker...')
            return
        if not self.tracking:
            self.get_logger().info('Marker found -> tracking')
            self.q_cmd = self.estimate_q()
            self.tracking = True

        # [2] 역기구학으로 목표 관절각
        q_goal, D, ok = camera_ik_limited(
            self.p_marker, self.desired_distance, self.L1,
            self.cam_x, self.cam_y, self.cam_yaw, self.q_cmd, self.q1_lim, self.q2_lim)
        reachable = ok and abs(D - self.desired_distance) < 1e-6
        self.q_goal = q_goal

        # [3] 속도 제한을 두고 목표각으로 이동
        err = q_goal - self.q_cmd
        err[np.abs(err) < self.deadband] = 0.0
        q_dot = self.gain * err
        peak = np.max(np.abs(q_dot))
        if peak > self.max_joint_vel:
            q_dot *= self.max_joint_vel / peak
        q_new = self.q_cmd + q_dot * self.dt
        # 실제 위치보다 max_lead 이상 앞서지 않게 (정지 시 미끄러짐 감소).
        # 보드는 관절각을 (-pi, pi] 로 보고하므로 차이는 wrap 해서 계산한다.
        q_est = self.estimate_q()
        lead = np.array([wrap(q_new[0] - q_est[0]), wrap(q_new[1] - q_est[1])])
        self.q_cmd = q_est + np.clip(lead, -self.max_lead, self.max_lead)

        self.publish(self.q_cmd)
        self.commanded = True
        self.publish_viz(D)
        self.log_throttled(
            f'marker base=({self.p_marker[0]:+.3f},{self.p_marker[1]:+.3f}) '
            f'| IK goal q=({q_goal[0]:+.3f},{q_goal[1]:+.3f})'
            f'{"" if reachable else f" [target distance unreachable -> D={D:.2f}]"} '
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
