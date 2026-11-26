# 로봇 프로그래밍 텀프로젝트 – Two-link Visual Servoing

twolink 로봇 + RealSense 카메라로 ArUco 마커를 추종하는 ROS 2 Humble 프로젝트입니다.

| 패키지 | 역할 |
| --- | --- |
| `twolink_msgs` | 목표 명령 `CMD`, 상태 피드백 `XYpos` 메시지 |
| `twolink_control` | 로봇 시리얼 통신(`serial_node`), 키보드 목표각 입력(`command_node`) |
| `twolink_gazebo` | URDF, RViz·Gazebo 시각화, 실제 로봇 ↔ Gazebo 연동 |
| `marker_detection` | RealSense 영상에서 ArUco 마커 인식 → `/aruco/pose` 발행 |

## 실행 환경
- Ubuntu 22.04 + ROS 2 Humble (desktop)
- `colcon`, `rosdep` (`sudo apt install python3-colcon-common-extensions python3-rosdep`)

## 설치 및 빌드 (처음 한 번)

```bash
source /opt/ros/humble/setup.bash

# rosdep 최초 1회 초기화 (이미 했다면 생략)
sudo rosdep init
rosdep update

# 작업공간에 clone
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/wony046/robot_programing.git

# 의존성 설치 (Gazebo, ros2_control, realsense2_camera, OpenCV 등 자동 설치)
cd ~/ros2_ws
sudo apt update
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y

# 빌드
colcon build --symlink-install
source install/setup.bash
```

새 터미널마다 `source ~/ros2_ws/install/setup.bash` 가 필요합니다.
매번 하기 싫으면: `echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc`

> `ros2_ws/src` 에 같은 이름의 패키지(예: 예전에 받은 `twolink_gazebo`)가 또 있으면
> colcon 이 "Duplicate package names" 오류를 냅니다. 중복 폴더를 지우고 빌드하세요.

## 실행

### 1. URDF 확인 (RViz만)
```bash
ros2 launch twolink_gazebo gazebo.launch.py use_gazebo:=false
ros2 run joint_state_publisher_gui joint_state_publisher_gui   # 다른 터미널, 슬라이더로 관절 이동
```

### 2. 실제 로봇과 Gazebo 연동
각 명령은 별도 터미널에서 실행합니다.
```bash
ros2 launch twolink_gazebo gazebo.launch.py
ros2 run twolink_control serial_node --ros-args -p port:=/dev/ttyACM0
ros2 run twolink_gazebo joint_publisher.py
```
- 포트는 `ls /dev/ttyACM*` 로 확인 (USB 를 다시 꽂으면 번호가 바뀔 수 있음)
- 권한 오류 시: `sudo usermod -aG dialout $USER` 후 재로그인
- 로봇 컨트롤 박스의 초록 시작 버튼을 눌러야 상태가 전송됩니다

실제 로봇 없이 테스트하려면 `serial_node` 대신:
```bash
ros2 run twolink_gazebo dummy_data_publisher.py
```

### 3. 키보드로 로봇 제어
```bash
ros2 run twolink_control serial_node --ros-args -p port:=/dev/ttyACM0
ros2 run twolink_control command_node      # e/d: 관절1 +/-, r/f: 관절2 +/-
```

### 4. 마커 추종 (Visual Servoing)
마커를 카메라 영상 좌우 중앙에 두고, 마커와 일정 거리를 유지하도록 로봇을 움직입니다.
```bash
ros2 launch twolink_control visual_servo.launch.py port:=/dev/ttyACM0 distance:=0.30
```
RealSense, 마커 인식, 시리얼, 시각 서보 노드와 결과 영상 뷰어(`/aruco/image`)가 한 번에 실행됩니다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `port` | `/dev/ttyACM0` | 로봇 시리얼 포트 |
| `distance` | `0.30` | 유지할 마커 거리 [m] |
| `target_id` | `-1` | 추종할 마커 ID (-1: 가장 크게 보이는 마커) |
| `dry_run` | `false` | `true`면 로봇에 명령을 보내지 않고 계산만 (처음 방향 확인용) |

- 관절이 수평 회전 2개뿐이라 좌우 중앙 정렬과 거리 유지만 제어합니다. 상하 정렬은 하지 않습니다.
- 마커를 놓치면 그 자리에 정지합니다. 팔이 닿지 않는 위치에서는 중앙 정렬을 우선합니다.
- 카메라 장착 위치(`camera_offset_x/y`, `camera_yaw`), 게인, 속도 제한은 `visual_servo_node` 파라미터로 조정합니다.

### 5. 마커 인식만 실행
```bash
ros2 launch realsense2_camera rs_launch.py
ros2 run marker_detection detection_realsense
```
- 입력: `/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`
- 출력: `/aruco/image` (인식 결과 영상), `/aruco/pose` (마커 자세), ArUco `DICT_4X4_50`

## 주요 토픽
| 토픽 | 타입 | 설명 |
| --- | --- | --- |
| `/robot_state` | `twolink_msgs/XYpos` | 실제(또는 dummy) 로봇 관절각 |
| `/des_value` | `twolink_msgs/CMD` | 로봇 목표값 |
| `/joint_states` | `sensor_msgs/JointState` | RViz 표시용 관절각 |
| `/aruco/pose` | `geometry_msgs/Pose` | 마커 자세 |
