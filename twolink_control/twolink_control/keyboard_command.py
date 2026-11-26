#!/usr/bin/env python3
import curses
import math
import rclpy
from rclpy.node import Node
from twolink_msgs.msg import CMD, XYpos
from geometry_msgs.msg import Point32

class PosPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_command')
        self.publisher = self.create_publisher(CMD, 'des_value', 10)
        self.q1_curr = 0.0
        self.q2_curr = 0.0
        self.delta = self.declare_parameter('delta', 0.02).value
        self.q1des = None
        self.q2des = None
        self.command_active = False
        # 현재 관절각 구독
        def feedback_callback(msg):
            self.q1_curr = msg.qpos1
            self.q2_curr = msg.qpos2
            # 첫 피드백으로 두 관절의 목표각을 초기화한다.
            if self.q1des is None and self.q2des is None:
                self.q1des = self.q1_curr
                self.q2des = self.q2_curr

        self.subscription = self.create_subscription(
            XYpos,
            'robot_state',
            feedback_callback,
            10
        )
        
        
        
        
    def update_position(self, stdscr):
        stdscr.nodelay(True)
        while rclpy.ok():
            key = stdscr.getch()
            if self.q1des is None or self.q2des is None:
                self.display_position(stdscr)
                stdscr.refresh()
                rclpy.spin_once(self, timeout_sec=0.01)
                continue
            self.handle_key(key)
            self.publish_message()
            self.display_position(stdscr)
            stdscr.refresh()
            rclpy.spin_once(self, timeout_sec=0.01)

    def handle_key(self, key):
        if self.q1des is None or self.q2des is None:
            return
        if key not in (ord('e'), ord('d'), ord('r'), ord('f')):
            return
        if not self.command_active:
            # Use the latest feedback when the first movement key is pressed.
            self.q1des = self.q1_curr
            self.q2des = self.q2_curr
        # Accumulate the target even when feedback lags or the joint is stationary.
        if key == ord('e'):
            self.q1des += self.delta
        elif key == ord('d'):
            self.q1des -= self.delta
        elif key == ord('r'):
            self.q2des += self.delta
        elif key == ord('f'):
            self.q2des -= self.delta
        self.command_active = True

    def display_position(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 0, f"Current q1={self.q1_curr:.4f}, q2={self.q2_curr:.4f}")
        # 표시만 current에 가장 가까운 동치 각도로 맞춘다 (2π 주기).
        q1des_display = self.q1_curr + (self.q1des - self.q1_curr + math.pi) % math.tau - math.pi if self.q1des is not None else None
        q2des_display = self.q2_curr + (self.q2des - self.q2_curr + math.pi) % math.tau - math.pi if self.q2des is not None else None
        q1des_str = f"{q1des_display:.4f}" if q1des_display is not None else "----"
        q2des_str = f"{q2des_display:.4f}" if q2des_display is not None else "----"
        stdscr.addstr(1, 0, f"Desired q1={q1des_str}, q2={q2des_str}")
        stdscr.addstr(3, 0, "Controls: e(+q1), d(-q1), r(+q2), f(-q2)")
        stdscr.addstr(4, 0, f"Step: {self.delta:.4f}")
        if self.q1des is None or self.q2des is None:
            stdscr.addstr(5, 0, "Waiting for robot_state...")
        elif not self.command_active:
            stdscr.addstr(5, 0, "Observing only: no joint command sent before a movement key.")

    def publish_message(self):
        if not self.command_active or self.q1des is None or self.q2des is None:
            # 아직 상태를 받지 못한 경우 publish를 건너뛴다
            return
        msg = CMD()
        msg.xdes = 0.0
        msg.ydes = 0.0
        msg.q1des = self.q1des
        msg.q2des = self.q2des
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PosPublisher()
    try:
        curses.wrapper(node.update_position)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
