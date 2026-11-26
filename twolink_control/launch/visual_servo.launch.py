"""마커 추종 전체 실행: RealSense + 마커 인식 + 시리얼 + 시각 서보
                     + 결과 영상 뷰어 + RViz/Gazebo 시각화

ros2 launch twolink_control visual_servo.launch.py distance:=0.30
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
        DeclareLaunchArgument('port', default_value='auto',
                              description='auto: /dev/twolink -> Arduino 자동 탐지'),
        DeclareLaunchArgument('distance', default_value='0.30',
                              description='마커와 유지할 거리 [m]'),
        DeclareLaunchArgument('target_id', default_value='-1',
                              description='추종할 마커 ID (-1: 가장 크게 보이는 마커)'),
        DeclareLaunchArgument('max_joint_vel', default_value='3.0',
                              description='최대 관절 속도 [rad/s]'),
        DeclareLaunchArgument('max_lead', default_value='0.8',
                              description='목표각 선행량 [rad] (클수록 빠름, 정지 시 더 미끄러짐)'),
        DeclareLaunchArgument('gain', default_value='10.0', description='추종 게인 [1/s]'),
        DeclareLaunchArgument('facing_weight', default_value='1.0',
                              description='마커 정면 각도 비중 (0: 거리만)'),
        DeclareLaunchArgument('elbow', default_value='positive',
                              description='역기구학 해 선택: positive / negative / nearest'),
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='true: 로봇에 명령을 보내지 않고 계산만'),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('color_profile', default_value='960x540x60',
                              description='RealSense 컬러 해상도x fps. 16:9(1920x1080/1280x720/960x540)는 '
                                          '원래 화각(약 69도), 640x480 같은 4:3은 좌우가 잘려 좁아짐'),
        DeclareLaunchArgument('use_serial', default_value='true'),
        DeclareLaunchArgument('use_viewer', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='RViz 로봇 모델 시각화 (robot_state 연동)'),
        DeclareLaunchArgument('use_gazebo', default_value='true',
                              description='Gazebo 시각화 (use_rviz:=true 일 때)'),

        # RViz(+Gazebo): twolink_gazebo 의 launch 재사용, joint_publisher 가 실제 관절각을 반영
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('twolink_gazebo'), 'launch', 'gazebo.launch.py'])),
            launch_arguments={'use_gazebo': LaunchConfiguration('use_gazebo')}.items(),
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
        Node(
            package='twolink_gazebo', executable='joint_publisher.py',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            output='log',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
            launch_arguments={
                'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
                'enable_depth': 'false',   # 마커로 거리를 추정하므로 depth 불필요 (USB/CPU 절약)
                'enable_infra1': 'false',
                'enable_infra2': 'false',
            }.items(),
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
            parameters=[{'desired_distance': distance, 'dry_run': dry_run,
                         'elbow': LaunchConfiguration('elbow'),
                         'max_joint_vel': LaunchConfiguration('max_joint_vel'),
                         'max_lead': LaunchConfiguration('max_lead'),
                         'gain': LaunchConfiguration('gain'),
                         'facing_weight': LaunchConfiguration('facing_weight')}],
            output='screen',
        ),
        Node(
            package='rqt_image_view', executable='rqt_image_view',
            arguments=['/aruco/image'],
            condition=IfCondition(LaunchConfiguration('use_viewer')),
        ),
    ])
