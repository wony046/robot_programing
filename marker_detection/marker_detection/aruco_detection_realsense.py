import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class ArucoDetectorNode_Realsense(Node):
    def __init__(self):
        super().__init__('aruco_detector_node_realsense')
        self.bridge = CvBridge()

        # 카메라 파라미터 저장
        self.camera_matrix = None
        self.dist_coeffs = None

        # 아루코 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.06 # m
        # 추종할 마커 ID (-1이면 가장 크게 보이는 마커)
        self.target_id = self.declare_parameter('target_id', -1).value
        # 시각화용 목표 거리 (visual_servo의 desired_distance와 맞춤)
        self.desired_distance = self.declare_parameter('desired_distance', 0.30).value

        # Subscriber
        # 항상 최신 프레임 1장만 처리 (처리가 밀려도 오래된 프레임이 쌓이지 않게)
        latest_only = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                                 reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            latest_only
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher
        self.image_pub = self.create_publisher(Image, '/aruco/image', 10)
        self.pose_pub = self.create_publisher(Pose, '/aruco/pose', 10)
        # 매 프레임 인식 여부 (False 가 오면 추종 노드가 즉시 정지)
        self.detected_pub = self.create_publisher(Bool, '/aruco/detected', 10)

        self.get_logger().info('ArucoDetectorNode start ~~ (Realsense)')

    def camera_info_callback(self, msg): # CameraInfo를 받으면 실행되는 callback 함수
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            self.camera_matrix = K
            self.dist_coeffs = D
            self.get_logger().info('CameraInfo OK ~~')

    def image_callback(self, msg): # Image를 받는면 실행되는 callback 함수
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        # ROS Image -> OpenCV 이미지
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 아루코 마커 검출
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        h, w = frame.shape[:2]
        # 기준 세로선: 영상 정중앙(광학 중심 cx). 로봇은 높이를 못 바꾸므로
        # 마커의 상하 위치는 무시하고 이 선과의 좌우 거리만 제어에 쓴다.
        fx = self.camera_matrix[0, 0]
        cx = self.camera_matrix[0, 2]
        line_x = int(round(cx))
        cv2.line(frame, (line_x, 0), (line_x, h - 1), (0, 255, 255), 2)

        result = None
        if ids is not None and len(ids) > 0: # 마커 검출에 성공했다면?
            # 마커 3차원 위치 추정
            result = self.my_estimatePoseSingleMarkers(corners, ids)

        if result is not None:
            rvec, tvec, quaternion, idx = result

             # 마커 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # 좌표축 그리기 (x: 빨강, y: 초록, z: 파랑)
            cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

            tvec = tvec.flatten()

            # 마커 중심 (렌즈 왜곡 보정한 픽셀 좌표)
            c_px = corners[idx][0].mean(axis=0)
            c_und = cv2.undistortPoints(c_px.reshape(1, 1, 2).astype(np.float64),
                                        self.camera_matrix, self.dist_coeffs,
                                        P=self.camera_matrix).reshape(2)
            # 세로선과의 좌우 거리: 픽셀 -> 마커 거리(Z)에서의 실제 거리 [m]
            dx_px = float(c_und[0] - cx)
            lateral = tvec[2] * dx_px / fx

            # 마커 중심 -> 세로선까지 수평 오차 표시 (상하는 무시)
            u, v = int(c_px[0]), int(c_px[1])
            cv2.line(frame, (line_x, v), (u, v), (0, 0, 255), 3)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID {int(ids[idx][0])}  L/R {lateral*100:+.1f} cm ({dx_px:+.0f} px)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Dist {tvec[2]*100:.1f} cm  (target {self.desired_distance*100:.0f} cm)',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 마커 정면 각도 psi: 마커 법선 n = R[:, 2], 정면이면 n = (0, 0, -1)
            n = cv2.Rodrigues(rvec)[0][:, 2]
            if n[2] < 0:
                psi = np.degrees(np.arctan2(-n[0], -n[2]))
                cv2.putText(frame, f'Facing {psi:+.0f} deg  (0 = parallel)',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 결과 Publish
            pose_msg = Pose()
            pose_msg.position.x = lateral    # 세로선 기준 좌우 거리 [m]
            pose_msg.position.y = tvec[1]
            pose_msg.position.z = tvec[2]
            pose_msg.orientation.x = quaternion[0]
            pose_msg.orientation.y = quaternion[1]
            pose_msg.orientation.z = quaternion[2]
            pose_msg.orientation.w = quaternion[3]
            self.pose_pub.publish(pose_msg)
            self.detected_pub.publish(Bool(data=True))

            self.get_logger().info(
                f'marker L/R={lateral:+.3f} m ({dx_px:+.0f} px) dist={tvec[2]:.3f} m',
                throttle_duration_sec=0.5)
        else:
            self.detected_pub.publish(Bool(data=False))
            cv2.putText(frame, 'No marker -> STOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # OpenCV -> ROS Image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        # 결과 이미지 Publish
        self.image_pub.publish(out_msg)

    def my_estimatePoseSingleMarkers(self, corners, ids):
        # 마커 좌표
        object_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],   # 좌상단 좌표
            [self.marker_length / 2, self.marker_length / 2, 0],    # 우상단 좌표
            [self.marker_length / 2, -self.marker_length / 2, 0],   # 우하단 좌표
            [-self.marker_length / 2, -self.marker_length / 2, 0]   # 좌하단 좌표
        ], dtype=np.float32)

        if ids is None:
            return None

        # 추종 대상 선택: target_id가 지정되면 그 마커, 아니면 가장 크게 보이는(가까운) 마커
        candidates = [i for i in range(len(ids))
                      if self.target_id < 0 or int(ids[i][0]) == self.target_id]
        candidates.sort(key=lambda i: -cv2.contourArea(corners[i][0].astype(np.float32)))

        for i in candidates:
            corner = corners[i][0]

            # solvePnP를 사용하여 3차원 위치 추정
            # ITERATIVE: 이 OpenCV(4.5.4)에서 IPPE 계열은 기울어진 마커의 법선을 뒤집어
            # 내는 경우가 있어 사용하지 않는다 (마커 각도 오차 최대 약 1도로 확인)
            success, rvec, tvec = cv2.solvePnP(
                object_points, corner, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if success and tvec[2, 0] > 0:
                quaternion = Rotation.from_rotvec(rvec.flatten()).as_quat() # xyzw
                return rvec, tvec, quaternion, i
        return None

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode_Realsense()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

