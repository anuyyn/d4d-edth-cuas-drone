#!/usr/bin/env python3

import argparse
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[-] ROS/cv_bridge 없음. 대체 방법 시도 중...")


class ObservationRecorder:
    def __init__(self, duration=30, output="observation.mp4", fps=30):
        self.duration = duration
        self.output = output
        self.fps = fps
        self.frames = []
        self.frame_count = 0
        self.max_frames = duration * fps
        self.temp_dir = None
        self.bridge = CvBridge() if HAS_ROS else None
        self.recording = False

    def start(self):
        if not HAS_ROS:
            print("[-] ROS 환경 필요. WSL에서 source /opt/ros/*/setup.bash 후 실행")
            return False

        print(f"[*] gzserver 시작 (world: balloon_observation)...")
        self.gz_proc = subprocess.Popen(
            ["gzserver", "--verbose", "worlds/balloon_observation.world"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)

        rospy.init_node("observation_recorder", anonymous=True)
        rospy.Subscriber(
            "/observation/up_camera/image",
            Image,
            self.image_callback
        )

        print("[+] ROS 노드 시작, 카메라 이미지 대기...")
        self.recording = True
        self.start_time = time.time()
        return True

    def image_callback(self, msg):
        if not self.recording or self.frame_count >= self.max_frames:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.frames.append(cv_image)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                print(f"[+] {self.frame_count}/{self.max_frames} frames ({elapsed:.1f}s)")

        except Exception as e:
            print(f"[-] 이미지 콜백 오류: {e}")

    def wait_and_record(self):
        print(f"[*] {self.duration}초 동안 녹화 중...")

        while self.frame_count < self.max_frames and self.recording:
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration:
                break
            rospy.sleep(0.01)

        self.recording = False
        print(f"[+] 녹화 완료: {len(self.frames)} frames 캡처")

    def encode_mp4(self):
        if not self.frames:
            print("[-] 캡처된 프레임 없음")
            return False

        print(f"[*] MP4 인코딩 중 ({len(self.frames)} frames)...")

        h, w = self.frames[0].shape[:2]
        out = cv2.VideoWriter(
            self.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h)
        )

        for i, frame in enumerate(self.frames):
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

            if (i + 1) % 30 == 0:
                print(f"[+] {i + 1}/{len(self.frames)} encoded...")

        out.release()
        print(f"[+] MP4 저장: {self.output}")
        return True

    def cleanup(self):
        if hasattr(self, 'gz_proc'):
            print("[*] gzserver 종료...")
            self.gz_proc.terminate()
            try:
                self.gz_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gz_proc.kill()

        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)


def main():
    parser = argparse.ArgumentParser(description="풍선 관찰 시나리오 녹화")
    parser.add_argument("--duration", type=int, default=30, help="녹화 시간 (초)")
    parser.add_argument("--output", type=str, default="videos/observation.mp4", help="출력 MP4 경로")
    parser.add_argument("--fps", type=int, default=30, help="FPS (기본 30)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    recorder = ObservationRecorder(
        duration=args.duration,
        output=args.output,
        fps=args.fps
    )

    try:
        if not recorder.start():
            return 1

        recorder.wait_and_record()
        recorder.encode_mp4()
        return 0

    except KeyboardInterrupt:
        print("\n[!] 중단됨")
        return 1

    except Exception as e:
        print(f"[-] 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        recorder.cleanup()


if __name__ == "__main__":
    sys.exit(main())
