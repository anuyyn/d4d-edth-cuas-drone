# D4D x EDTH 2026 온디바이스 AI 정찰드론 (T1 UAS/CUAS)

Raspberry Pi Zero 2 W 정찰드론이 카메라 영상에서 드론을 탐지하고, 고글 화면과 노트북 관제 GUI에 박스와 경보를 표시합니다.
요격과 부착 추적은 Gazebo 시뮬레이션으로 시연합니다.

## 폴더
- cuas/detect_only : 최종 시연 코드. 파이 탐지(pi_fb_detect.py), 관제 GUI(gui/detect_gui.py), SD카드 설치(write_sd.py)
- cuas/appearance : 외형 분류기 학습(MobileNetV3-Small, 128px, ONNX)
- cuas : 움직임 기반 검출/추적(motion_id.py) 및 이전 버전
- cuas/hw : 하드웨어 구성도, 비행 컨트롤러 회로도(Hackathon.pdf)
- sim : Gazebo A.R.M.S. 요격/부착 시뮬레이션
- trainer : FPV 훈련기와 IMU 컨트롤러 펌웨어
- docs : 시연 계획(계획.pdf)

## 실행
    export CUAS_PI_USER=pi CUAS_PI_PASS=<password>
    python cuas/detect_only/run_detect_only.py
    python cuas/detect_only/write_sd.py D:/

## 데이터
학습 사진, 탐지 로그, 영상은 포함하지 않았습니다. 샘플 사진 3장만 cuas/appearance/sample_photos 에 있습니다.
공개 데이터셋: eashmanova/Drons_and_Birds_Detection, ImageNet balloon(mlnomad/imnet1k_balloon)
