"""Check the serial contract used by arduino/simplerobot_control.ino."""

from collections import deque
import struct
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from twolink_control import serial_interface


# Actual 32-byte board response captured while debugging this connection.
STATE_SAMPLE = bytes.fromhex(
    '1d 1e 51 bd 3d b2 50 3d 72 5b f4 b9 8e b1 50 39 '
    '48 28 46 40 2a 4a 29 c0 00 10 a0 3b 00 10 a0 3b')


def make_bridge():
    """Exercise callbacks without starting ROS or opening a hardware port."""
    logger = Mock()
    bridge = SimpleNamespace(
        xdes=0.0, ydes=0.0, q1des=0.0, q2des=0.0,
        command_received=False, state_lock=threading.Lock(),
        stop_event=threading.Event(), last_state=None, rx_bytes=0,
        rx_packets=0, last_packet_time=None, rx_preview=b'',
        rx_preview_reported=False, last_published_packet=0,
        pending_request=None, pending_request_time=0.0, board_diagnostics=True,
        next_diagnostic_time=float('inf'), last_board_diagnostic=None,
        last_board_diagnostic_time=None,
        pub_robot_state=Mock(), get_logger=lambda: logger)
    for name in ('build_packet', 'parse_packet', 'listener_callback',
                 'rx_loop', 'handle_response', 'tx_and_publish', 'report_serial_status'):
        setattr(bridge, name, MethodType(
            getattr(serial_interface.SerialSubscriber, name), bridge))
    return bridge


def test_startup_only_requests_state():
    assert make_bridge().build_packet() == b'R'


def test_joint_command_matches_firmware_and_requests_feedback():
    bridge = make_bridge()
    bridge.listener_callback(SimpleNamespace(
        xdes=0.0, ydes=0.0, q1des=3.0, q2des=-2.5))
    packet = bridge.build_packet()

    # Firmware accepts J followed by exactly 8 floats; a separate non-C/J
    # byte is needed to set sendflag after completing the command.
    assert len(packet) == 34
    assert packet[:1] == b'J'
    assert struct.unpack('<8f', packet[1:33]) == (
        0.0, 0.0, 0.0, 0.0, 3.0, -2.5, 0.0, 0.0)
    assert packet[33:] == b'R'


def test_decode_real_board_capture():
    state = make_bridge().parse_packet(STATE_SAMPLE)
    assert state == pytest.approx((
        -0.0510541089, 0.0509512313, -0.0004660744, 0.0001990257,
        3.0962085724, -2.6451516151, 0.0048847198, 0.0048847198))


@pytest.mark.parametrize('packet', [
    b'', STATE_SAMPLE[:-1], STATE_SAMPLE + b'\0',
    struct.pack('<8f', float('nan'), 0, 0, 0, 0, 0, 0, 0),
    struct.pack('<8f', 0, 0, 0, 0, float('inf'), 0, 0, 0),
])
def test_reject_incomplete_or_nonfinite_state(packet):
    assert make_bridge().parse_packet(packet) is None


def test_nonfinite_command_does_not_enable_joint_control():
    bridge = make_bridge()
    bridge.listener_callback(SimpleNamespace(
        xdes=0.0, ydes=0.0, q1des=float('nan'), q2des=0.0))
    assert bridge.build_packet() == b'R'


def test_rx_preserves_split_and_combined_frames(monkeypatch):
    bridge = make_bridge()
    frames = []
    bridge.handle_response = frames.append
    second = struct.pack('<8f', 0, 0, 0, 0, 1.0, -1.0, 0, 0)
    chunks = deque([
        STATE_SAMPLE[:5], STATE_SAMPLE[5:19],
        STATE_SAMPLE[19:] + second + STATE_SAMPLE[:7],
        STATE_SAMPLE[7:],
    ])

    class FakeSerial:
        @property
        def in_waiting(self):
            return len(chunks[0]) if chunks else 0

        def read(self, count):
            if not chunks:
                bridge.stop_event.set()
                return b''
            chunk = chunks.popleft()
            assert count == len(chunk)
            return chunk

    bridge.ser = FakeSerial()
    monkeypatch.setattr(serial_interface.rclpy, 'ok', lambda: True)
    bridge.rx_loop()
    assert frames == [STATE_SAMPLE, second, STATE_SAMPLE]
    assert bridge.rx_bytes == 96


def test_only_new_feedback_is_published():
    bridge = make_bridge()
    bridge.ser = Mock()
    bridge.last_state = bridge.parse_packet(STATE_SAMPLE)
    bridge.rx_packets = 1
    bridge.tx_and_publish()
    bridge.tx_and_publish()

    bridge.pub_robot_state.publish.assert_called_once()
    message = bridge.pub_robot_state.publish.call_args.args[0]
    assert message.qpos1 == pytest.approx(3.0962085724)
    assert message.qpos2 == pytest.approx(-2.6451516151)
    assert all(call.args == (b'R',) for call in bridge.ser.write.call_args_list)

    bridge.rx_packets = 2
    bridge.tx_and_publish()
    assert bridge.pub_robot_state.publish.call_count == 2


def diagnostic_packet():
    # status1=1, enable requested/brake/seen, CAN age=3ms.
    # status2=0xD, enable requested/brake/seen, CAN age=1200ms.
    motor1 = 1 | (1 << 4) | (1 << 5) | (1 << 6) | (3 << 8)
    motor2 = 0xD | (1 << 4) | (1 << 5) | (1 << 6) | (1200 << 8)
    flags = ord('J') | (1 << 8) | (1 << 9)
    return struct.pack('<8f', 12345.0, flags, 3.199, -2.6409,
                       1.7, -0.12, motor1, motor2)


def test_board_diagnostic_fields_and_can_freshness():
    diagnostic = serial_interface.decode_board_diagnostics(diagnostic_packet())
    assert diagnostic['mode'] == 'J'
    assert diagnostic['running'] and diagnostic['relay_pin']
    assert diagnostic['targets'] == pytest.approx((3.199, -2.6409))
    assert diagnostic['torques'] == pytest.approx((1.7, -0.12))
    assert diagnostic['motors'][0] == {
        'status': 1, 'enable_requested': True, 'brake_pin': True,
        'can_seen': True, 'locked': False, 'can_age_ms': 3,
    }
    assert diagnostic['motors'][1]['status'] == 0xD
    assert diagnostic['motors'][1]['can_age_ms'] == 1200


def test_diagnostic_reply_is_not_published_as_joint_positions():
    bridge = make_bridge()
    bridge.pending_request = b'D'
    bridge.handle_response(diagnostic_packet())
    assert bridge.last_board_diagnostic['mode'] == 'J'
    assert bridge.pending_request is None
    assert bridge.last_state is None
    assert bridge.rx_packets == 0
    bridge.ser = Mock()
    bridge.tx_and_publish()
    bridge.pub_robot_state.publish.assert_not_called()


def test_delayed_diagnostic_is_not_mistaken_for_state():
    bridge = make_bridge()
    bridge.pending_request = b'R'
    bridge.handle_response(diagnostic_packet())
    assert bridge.last_state is None
    assert bridge.rx_packets == 0
    assert bridge.pending_request is None


def test_old_firmware_disables_diagnostics_and_continues_state_polling():
    bridge = make_bridge()
    bridge.pending_request = b'D'
    # Older firmware treats D as an ordinary R and returns positions.
    bridge.handle_response(STATE_SAMPLE)
    assert not bridge.board_diagnostics
    assert bridge.last_state is None
    assert bridge.pending_request is None
    bridge.ser = Mock()
    bridge.next_diagnostic_time = 0
    bridge.rx_packets = 1
    bridge.tx_and_publish()
    bridge.ser.write.assert_called_once_with(b'R')
    bridge.handle_response(STATE_SAMPLE)
    assert bridge.last_state == bridge.parse_packet(STATE_SAMPLE)


def test_no_second_request_until_full_response():
    bridge = make_bridge()
    bridge.ser = Mock()
    bridge.rx_packets = 1
    bridge.next_diagnostic_time = 0
    bridge.tx_and_publish()
    bridge.tx_and_publish()
    bridge.ser.write.assert_called_once_with(b'D')
    bridge.handle_response(diagnostic_packet())
    bridge.tx_and_publish()
    assert [call.args[0] for call in bridge.ser.write.call_args_list] == [b'D', b'R']


def test_joint_command_can_request_board_diagnostics():
    bridge = make_bridge()
    bridge.listener_callback(SimpleNamespace(
        xdes=0.0, ydes=0.0, q1des=3.199, q2des=-2.6409))
    packet = bridge.build_packet(b'D')
    assert len(packet) == 34
    assert packet[:1] == b'J' and packet[-1:] == b'D'
    assert struct.unpack('<8f', packet[1:33])[4:6] == pytest.approx((3.199, -2.6409))


def test_lost_reply_retries_same_request_without_changing_response_type(monkeypatch):
    bridge = make_bridge()
    bridge.ser = Mock()
    monkeypatch.setattr(serial_interface.time, 'monotonic', lambda: 1.0)
    bridge.tx_and_publish()
    assert bridge.pending_request == b'R'
    bridge.next_diagnostic_time = 0
    monkeypatch.setattr(serial_interface.time, 'monotonic', lambda: 1.6)
    bridge.tx_and_publish()
    assert [call.args[0] for call in bridge.ser.write.call_args_list] == [b'R', b'R']
    bridge.handle_response(STATE_SAMPLE)
    bridge.tx_and_publish()
    assert bridge.ser.write.call_args.args == (b'D',)


def test_stopped_board_remains_observable_before_first_state():
    bridge = make_bridge()
    bridge.ser = Mock()
    bridge.next_diagnostic_time = 0
    bridge.tx_and_publish()
    bridge.ser.write.assert_called_once_with(b'D')
    stopped = struct.pack('<8f', 12345, ord('C'), 0, 0, 0, 0, 65535 << 8, 65535 << 8)
    bridge.handle_response(stopped)
    bridge.tx_and_publish()
    assert [call.args[0] for call in bridge.ser.write.call_args_list] == [b'D', b'D']
    bridge.handle_response(diagnostic_packet())  # board is now running
    bridge.tx_and_publish()
    assert bridge.ser.write.call_args.args == (b'R',)
    bridge.pub_robot_state.publish.assert_not_called()


@pytest.mark.parametrize('packet', [
    STATE_SAMPLE, diagnostic_packet()[:-1],
    struct.pack('<8f', 12345, ord('X'), 0, 0, 0, 0, 0, 0),
    struct.pack('<8f', 12345, ord('J'), 0, 0, float('nan'), 0, 0, 0),
    struct.pack('<8f', 12345, ord('J'), 0, 0, 0, 0, -1, 0),
    struct.pack('<8f', 12345, ord('J'), 0, 0, 0, 0, 1.5, 0),
])
def test_reject_invalid_diagnostic_records(packet):
    assert serial_interface.decode_board_diagnostics(packet) is None
