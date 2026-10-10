#!/bin/bash

cd "$(dirname "$0")" 2>/dev/null || exit 1
mkdir -p videos logs

DURATION=${1:-30}
VIDEO="videos/observation_${DURATION}s.mp4"
LOG="logs/rec_observation.csv"

echo "[*] Gazebo 풍선 관찰 시뮬 시작 (${DURATION}초)..."


if command -v gazebo &> /dev/null; then
    gzserver worlds/balloon_observation.world &
    GZ_PID=$!

    sleep 3

    python3 << 'EOF'
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

bridge = CvBridge()
frames = []
recording = True
frame_count = 0
max_frames = int(30 * 30)

def image_callback(msg):
    global frames, frame_count, recording
    if not recording or frame_count >= max_frames:
        return
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        frames.append(cv_image)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[+] {frame_count} frames captured...")
    except Exception as e:
        print(f"Error: {e}")

try:
    rospy.init_node('balloon_recorder', anonymous=True)
    rospy.Subscriber('/observation/up_camera/image', Image, image_callback)

    print("[+] ROS 노드 시작, 카메라 토픽 대기...")
    rospy.sleep(2)

    start_time = rospy.get_time()
    while len(frames) < max_frames and rospy.get_time() - start_time < 30:
        rospy.sleep(0.01)

    recording = False

    if frames:
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(
            'videos/observation.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30, (w, h)
        )

        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()
        print(f"[+] MP4 저장: videos/observation.mp4 ({len(frames)} frames)")
    else:
        print("[-] 캡처된 프레임 없음")

except Exception as e:
    print(f"[-] 오류: {e}")
finally:
    rospy.signal_shutdown('Done')

EOF

    kill $GZ_PID 2>/dev/null
    wait $GZ_PID 2>/dev/null

else
    echo "[-] gazebo 명령어 없음. WSL에서 실행 필요"
    exit 1
fi

echo "[*] 완료: $VIDEO"
