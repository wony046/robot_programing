"""마커 추종 전체 실행: RealSense + 마커 인식 + 시리얼 + 시각 서보 (+ 결과 영상 뷰어)

ros2 launch twolink_control visual_servo.launch.py port:=/dev/ttyACM0 distance:=0.30
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    port = LaunchConfiguration('port')
    distance = LaunchConfiguration('distance')
    dry_run = LaunchConfiguration('dry_run')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('distance', default_value='0.30',
                              description='마커와 유지할 거리 [m]'),
        DeclareLaunchArgument('target_id', default_value='-1',
                              description='추종할 마커 ID (-1: 가장 크게 보이는 마커)'),
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='true: 로봇에 명령을 보내지 않고 계산만'),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_serial', default_value='true'),
        DeclareLaunchArgument('use_viewer', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
            condition=IfCondition(LaunchConfiguration('use_camera')),
        ),
        Node(
            package='marker_detection', executable='detection_realsense',
            parameters=[{'target_id': LaunchConfiguration('target_id'),
                         'desired_distance': distance}],
            output='screen',
        ),
        Node(
            package='twolink_control', executable='serial_node',
            parameters=[{'port': port}],
            condition=IfCondition(LaunchConfiguration('use_serial')),
            output='screen',
        ),
        Node(
            package='twolink_control', executable='visual_servo_node',
            parameters=[{'desired_distance': distance, 'dry_run': dry_run}],
            output='screen',
        ),
        Node(
            package='rqt_image_view', executable='rqt_image_view',
            arguments=['/aruco/image'],
            condition=IfCondition(LaunchConfiguration('use_viewer')),
        ),
    ])
