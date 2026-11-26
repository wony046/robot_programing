#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from twolink_msgs.msg import XYpos, CMD
import serial
import struct
import sys
import threading
import time


def decode_board_diagnostics(data):
    """Decode a D response; these values must never become robot_state."""
    if len(data) != 32:
        return None
    values = struct.unpack('<8f', data)
    if not all(math.isfinite(value) for value in values) or values[0] != 12345.0:
        return None
    flags = values[1]
    if not flags.is_integer() or not 0 <= flags <= 1023:
        return None
    flags = int(flags)
    if flags & 0xFF not in (ord('C'), ord('J')):
        return None
    motors = []
    for value in values[6:8]:
        if not value.is_integer() or not 0 <= value <= 0xFFFFFF:
            return None
        word = int(value)
        motors.append({
            'status': word & 0xF,
            'enable_requested': bool(word & (1 << 4)),
            'brake_pin': bool(word & (1 << 5)),
            'can_seen': bool(word & (1 << 6)),
            'locked': bool(word & (1 << 7)),
            'can_age_ms': word >> 8,
        })
    return {
        'mode': chr(flags & 0xFF),
        'running': bool(flags & (1 << 8)),
        'relay_pin': bool(flags & (1 << 9)),
        'targets': values[2:4],
        'torques': values[4:6],
        'motors': motors,
    }


class SerialSubscriber(Node):
    def __init__(self):
        super().__init__('serial_interface')
        self.pub_robot_state = self.create_publisher(XYpos, "robot_state", 10)
        self.subscription = self.create_subscription(
            CMD, 'des_value', self.listener_callback, 10)

        # 명령 값
        self.xdes = 0.0
        self.ydes = 0.0
        self.q1des = 0.0
        self.q2des = 0.0
        self.command_received = False

        # 수신된 로봇 상태 (RX 스레드가 갱신)
        self.last_state = None
        self.state_lock = threading.Lock()
        self.rx_bytes = 0
        self.rx_packets = 0
        self.last_packet_time = None
        self.rx_preview = b''
        self.rx_preview_reported = False
        self.last_published_packet = 0
        self.stop_event = threading.Event()
        self.pending_request = None
        self.pending_request_time = 0.0
        self.last_board_diagnostic = None
        self.last_board_diagnostic_time = None
        self.board_diagnostics = self.declare_parameter('board_diagnostics', True).value
        # Probe D first: updated firmware can report why a stopped board is
        # not answering R. Older firmware falls back to R after its first reply.
        self.next_diagnostic_time = 0.0

        # 34-byte command/request + 32-byte response fit within 9600 baud at 10 Hz.
        pub_period = 0.1
        port = self.declare_parameter('port', '/dev/ttyACM0').value
        baudrate = self.declare_parameter('baudrate', 9600).value

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.01, write_timeout=0.2)
            # Let an old partial command expire (firmware timeout: 100 ms), then
            # discard pending replies before requesting fresh, headerless frames.
            time.sleep(0.15)
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            sys.exit(1)

        # RX 스레드 시작
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.rx_thread.start()

        # 타이머: 주기적으로 TX + publish
        self.timer = self.create_timer(pub_period, self.tx_and_publish)
        self.diagnostic_timer = self.create_timer(2.0, self.report_serial_status)
        self.get_logger().info(
            f'Serial connected: {port}, {baudrate} baud; '
            'joint command J + 8 float32, state request R, diagnostics D, response 32 bytes')

    def report_serial_status(self):
        with self.state_lock:
            rx_bytes = self.rx_bytes
            rx_packets = self.rx_packets
            last_packet_time = self.last_packet_time
            rx_preview = self.rx_preview
            state = self.last_state
            board = self.last_board_diagnostic
            board_time = self.last_board_diagnostic_time

        if rx_bytes == 0:
            self.get_logger().warning(
                'RX: 0 bytes received. Firmware sends state only while running; '
                'check board run state, serial port and connection.')
        elif rx_packets == 0 and board is not None:
            self.get_logger().info(
                'Board diagnostics received; waiting for position feedback. '
                + ('Board is not running; the start button begins control.'
                   if not board['running'] else 'Waiting for an R response.'))
        elif rx_packets == 0:
            self.get_logger().warning(
                f'RX: {rx_bytes} bytes, 0 decoded frames. Expected 32 bytes: '
                '8 little-endian float32 values, without header or CRC. '
                'Check firmware, baud rate and frame alignment.')
            if not self.rx_preview_reported:
                self.get_logger().warning(f'RX sample HEX: {rx_preview.hex(" ")}')
                self.get_logger().warning(f'RX sample BYTES: {rx_preview!r}')
                self.rx_preview_reported = True
        else:
            age = time.monotonic() - last_packet_time
            target = (
                f'q1des={self.q1des:.4f}, q2des={self.q2des:.4f}'
                if self.command_received else 'waiting for /des_value')
            status = (
                f'RX: {rx_bytes} bytes, {rx_packets} decoded frames, last frame {age:.2f}s ago; '
                f'q1={state[4]:.4f}, q2={state[5]:.4f}; TX target: {target}')
            if age > 2.0:
                self.get_logger().warning(status)
            else:
                self.get_logger().info(status)

        if board is not None:
            motors = board['motors']
            motor_text = []
            for index, motor in enumerate(motors, 1):
                age = f"{motor['can_age_ms']}ms" if motor['can_seen'] else 'never'
                motor_text.append(
                    f"M{index}(status=0x{motor['status']:X}, CAN_age={age}, "
                    f"enable_requested={int(motor['enable_requested'])}, "
                    f"brake_pin={int(motor['brake_pin'])}, locked={int(motor['locked'])})")
            status = (
                f"BOARD: mode={board['mode']}, running={int(board['running'])}, "
                f"relay_pin={int(board['relay_pin'])}, "
                f"applied_q1des={board['targets'][0]:.4f}, "
                f"applied_q2des={board['targets'][1]:.4f}, "
                f"torque1={board['torques'][0]:.4f}, torque2={board['torques'][1]:.4f}; "
                + ' '.join(motor_text)
                + f'; report_age={time.monotonic() - board_time:.2f}s')
            unhealthy = (
                not board['running'] or not board['relay_pin']
                or time.monotonic() - board_time > 5.0
                or (self.command_received and board['mode'] != 'J')
                or any(m['status'] >= 8 or not m['can_seen']
                       or m['can_age_ms'] > 500 or m['locked'] for m in motors))
            if unhealthy:
                self.get_logger().warning(status)
            else:
                self.get_logger().info(status)

    # ---------------- Firmware protocol ----------------
    def build_packet(self, request=b'R') -> bytes:
        # Any non-C/J byte requests state when firmware is outside a payload.
        # Do not send zero joint targets before the first keyboard command.
        if not self.command_received:
            return request
        payload = struct.pack(
            '<8f',
            self.xdes, self.ydes, 0.0, 0.0,
            self.q1des, self.q2des, 0.0, 0.0
        )
        # A complete J command does not request feedback by itself.
        return b'J' + payload + request

    def parse_packet(self, data: bytes):
        # sendRobotState(): x, y, vx, vy, q1, q2, dq1, dq2 (no header/CRC).
        if len(data) != 32:
            return None
        values = struct.unpack('<8f', data)
        # A delayed diagnostic record must never be used as joint feedback,
        # even if it arrives while an R response is expected.
        if values[0] == 12345.0 or not all(math.isfinite(value) for value in values):
            return None
        return values

    def handle_response(self, packet):
        # Only one R/D request is outstanding, so a diagnostic reply cannot be
        # mistaken for position feedback. Keep the slot until decoding finishes.
        with self.state_lock:
            request = self.pending_request
        if request == b'D':
            diagnostic = decode_board_diagnostics(packet)
            with self.state_lock:
                if diagnostic is None:
                    self.board_diagnostics = False
                else:
                    self.last_board_diagnostic = diagnostic
                    self.last_board_diagnostic_time = time.monotonic()
                self.pending_request = None
            if diagnostic is None:
                self.get_logger().warning(
                    'Board diagnostic response not recognized. Upload the updated '
                    'sketch and DriveWrapper library. Continuing with R state requests.')
        elif request == b'R':
            values = self.parse_packet(packet)
            with self.state_lock:
                if values is not None:
                    self.last_state = values
                    self.rx_packets += 1
                    self.last_packet_time = time.monotonic()
                self.pending_request = None

    # ---------------- RX 스레드 ----------------
    def rx_loop(self):
        """
        시리얼 수신 전용 스레드:
        - 계속해서 바이트 스트림을 읽고
        - 분할 수신된 바이트를 모아 32바이트씩 해석
        - 헤더/CRC가 없으므로 전송 중 바이트 유실 시 자동 재동기화 불가
        """
        buf = bytearray()

        while rclpy.ok() and not self.stop_event.is_set():
            try:
                # No busy loop while idle; read one byte with the serial timeout.
                n = self.ser.in_waiting
                chunk = self.ser.read(max(1, n))
                if chunk:
                    with self.state_lock:
                        self.rx_bytes += len(chunk)
                        # 최근 원시 바이트를 보관한다. 패킷 경계와 일치하지 않을 수 있다.
                        self.rx_preview = (self.rx_preview + chunk)[-96:]
                    buf.extend(chunk)

                # 패킷 파싱 루프
                while len(buf) >= 32:
                    packet = bytes(buf[:32])
                    del buf[:32]

                    self.handle_response(packet)

            except Exception as e:
                if self.stop_event.is_set():
                    break
                self.get_logger().error(f"RX loop error: {e}")
                self.stop_event.wait(0.1)

    # ---------------- TX + Publish ----------------
    def tx_and_publish(self):
        # 1) Complete each reply before issuing a different request type.
        request = None
        now = time.monotonic()
        with self.state_lock:
            if self.pending_request is None:
                request = b'R'
                board_stopped = (
                    self.last_board_diagnostic is not None
                    and not self.last_board_diagnostic['running'])
                if self.board_diagnostics and (board_stopped or now >= self.next_diagnostic_time):
                    request = b'D'
                    self.next_diagnostic_time = now + 2.0
                self.pending_request = request
                self.pending_request_time = now
            elif now - self.pending_request_time >= 0.5:
                # Retry the SAME response type after a lost request/reply.
                # Changing R to D here would make headerless replies ambiguous.
                request = self.pending_request
                self.pending_request_time = now
        if request is not None:
            try:
                self.ser.write(self.build_packet(request))
            except Exception as e:
                with self.state_lock:
                    self.pending_request = None
                self.get_logger().error(f"TX error: {e}")
                return

        # 2) 가장 최근 상태값을 읽어와서 publish
        with self.state_lock:
            state = self.last_state
            packet_number = self.rx_packets

        if state is None or packet_number == self.last_published_packet:
            # Publish only newly received state, not stale cached feedback.
            return

        x, y, xv, yv, q1, q2, q1v, q2v = state

        self.get_logger().debug(
            f"x={x:.4f}, y={y:.4f}, xv={xv:.4f}, yv={yv:.4f}, "
            f"q1={q1:.4f}, q2={q2:.4f}, q1v={q1v:.4f}, q2v={q2v:.4f}"
        )

        msg = XYpos()
        msg.xpos = x
        msg.ypos = y
        msg.xvel = xv
        msg.yvel = yv
        msg.qpos1 = q1
        msg.qpos2 = q2
        msg.qvel1 = q1v
        msg.qvel2 = q2v
        self.pub_robot_state.publish(msg)
        self.last_published_packet = packet_number

    # ---------------- ROS 콜백 ----------------
    def listener_callback(self, cmd: CMD):
        values = (cmd.xdes, cmd.ydes, cmd.q1des, cmd.q2des)
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warning('Ignoring non-finite /des_value command')
            return
        self.xdes = cmd.xdes
        self.ydes = cmd.ydes
        self.q1des = cmd.q1des
        self.q2des = cmd.q2des
        self.command_received = True

    def destroy_node(self):
        self.stop_event.set()
        self.rx_thread.join(timeout=1.0)
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

