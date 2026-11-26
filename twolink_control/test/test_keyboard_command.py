"""Verify safe startup and cumulative keyboard joint targets."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from twolink_control.keyboard_command import PosPublisher


def keyboard_with_feedback():
    keyboard = SimpleNamespace(
        q1_curr=3.099, q2_curr=-2.6409, q1des=3.0, q2des=-2.5,
        delta=0.1, command_active=False, publisher=Mock())
    for name in ('handle_key', 'publish_message'):
        setattr(keyboard, name, MethodType(getattr(PosPublisher, name), keyboard))
    return keyboard


@pytest.mark.parametrize('key', [-1, ord('x'), ord(' '), ord('\n')])
def test_feedback_and_nonmovement_keys_do_not_start_joint_control(key):
    keyboard = keyboard_with_feedback()
    keyboard.handle_key(key)
    keyboard.publish_message()
    assert not keyboard.command_active
    keyboard.publisher.publish.assert_not_called()


def test_movement_key_requires_feedback():
    keyboard = keyboard_with_feedback()
    keyboard.q1des = keyboard.q2des = None
    keyboard.handle_key(ord('e'))
    keyboard.publish_message()
    assert not keyboard.command_active
    keyboard.publisher.publish.assert_not_called()


def test_first_movement_key_uses_latest_feedback_for_both_joints():
    keyboard = keyboard_with_feedback()
    keyboard.handle_key(ord('e'))
    keyboard.publish_message()
    assert keyboard.command_active
    keyboard.publisher.publish.assert_called_once()
    command = keyboard.publisher.publish.call_args.args[0]
    assert command.q1des == pytest.approx(3.199)
    assert command.q2des == pytest.approx(-2.6409)


@pytest.mark.parametrize('key, joint, direction', [
    ('e', 1, 1), ('d', 1, -1), ('r', 2, 1), ('f', 2, -1),
])
def test_repeated_keys_accumulate_when_feedback_is_stationary(key, joint, direction):
    keyboard = keyboard_with_feedback()
    initial_targets = (keyboard.q1_curr, keyboard.q2_curr)

    for count in range(1, 4):
        keyboard.handle_key(ord(key))
        keyboard.publish_message()

        expected = list(initial_targets)
        expected[joint - 1] += direction * count * keyboard.delta
        command = keyboard.publisher.publish.call_args.args[0]
        assert (command.q1des, command.q2des) == pytest.approx(expected)


def test_feedback_changes_and_switching_joints_preserve_accumulated_targets():
    keyboard = keyboard_with_feedback()
    # Feedback follows only partway. Changing joints must retain both targets.
    steps = [
        ('e', (3.099, -2.6409), (3.199, -2.6409)),
        ('e', (3.12, -2.63), (3.299, -2.6409)),
        ('r', (3.14, -2.62), (3.299, -2.5409)),
        ('d', (3.16, -2.61), (3.199, -2.5409)),
        ('f', (3.18, -2.60), (3.199, -2.6409)),
    ]
    for key, feedback, expected in steps:
        keyboard.q1_curr, keyboard.q2_curr = feedback
        keyboard.handle_key(ord(key))
        keyboard.publish_message()

        command = keyboard.publisher.publish.call_args.args[0]
        assert (command.q1des, command.q2des) == pytest.approx(expected)
