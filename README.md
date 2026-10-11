# D4D x EDTH 2026 온디바이스 AI 정찰드론 (T1 UAS/CUAS)

KOREAN DEFENSE TECH HACKATHON (D4D x EDTH, 2026.10.10~11, 서울) 출품작입니다.

시연 영상: https://anuyyn.github.io/d4d-edth-cuas-drone/

발표 자료: https://anuyyn.github.io/d4d-edth-cuas-drone/presentation.pdf

소형 정찰드론이 비행 중에 카메라 영상을 기체 안의 Raspberry Pi에서 직접 분석해 드론을 찾아냅니다.
찾은 결과는 조종사 고글 화면에 박스와 글자로 표시되고, 같은 시각 Wi-Fi로 관제 노트북에도 전송됩니다.
관제 근무자는 노트북 화면에서 탐지 내용을 확인한 뒤 대응을 지시하고, 대응(요격, 추적 태그 부착)은 Gazebo 시뮬레이션으로 보여줍니다.

서버나 클라우드 없이 드론 안에서 판단까지 끝내는 것이 목표입니다. 통신이 끊겨도 고글 화면의 탐지 표시는 계속 동작합니다.


## 전체 구성

```
[정찰드론]
  Pi Camera v2 ──CSI──> Raspberry Pi Zero 2 W
                          ├─ 표시 프로세스: 카메라 영상 + 박스 → 컴포지트 출력 → 아날로그 VTX(5.8GHz) → FPV 고글
                          └─ 분석 프로세스: 검출 + 외형 분류(ONNX) → 박스·판정
                                   │
                                   │ Wi-Fi UDP 9500 (영상 + 박스 + 판정)
                                   v
[관제 노트북]
  detect_gui.py : 실시간 영상, 박스, "드론 탐지되었습니다" 경보, 이벤트 로그, 사람 확인/정정
                   │ Enter = 대응 실행
                   │ UDP 9510 (intercept / track 명령, 상태 회신)
                   v
  A.R.M.S. 시뮬 (WSL, Gazebo Classic 11) : 요격 드론 발진 → 탐색 → 추적 → 명중
```

조종은 TX12 조종기와 ELRS 2.4GHz 수신기, 비행 컨트롤러는 DAKEFPV F4(Betaflight)를 씁니다.
Pi와 FC는 UART(MSP)로 연결되어 있지만, 최종 시연본은 비행 명령을 보내지 않고 탐지만 합니다(`--fc none`).


## 하드웨어

- 기체: 3D 프린팅 프레임, RS2205 모터
- 컴퓨터: Raspberry Pi Zero 2 W + Pi Camera v2
- 비행 컨트롤러: DAKEFPV F4, Betaflight
- 영상: Pi 컴포지트 출력 → 아날로그 VTX 5.8GHz → FPV 고글
- 조종: TX12 조종기, ELRS 2.4GHz
- 관제: Windows 노트북(WSL Ubuntu 22.04), 모든 무선 구간은 같은 핫스팟 사용
- 풍선 대응용 추적 드론: 카본 막대 40cm 끝에 V자 깔때기와 카라비너, 자석 연결부로 태그를 줄에 걸고 빠짐

자세한 연결은 `cuas/hw/hardware_diagram.png`, 팀에서 설계한 비행 컨트롤러 회로도(STM32F405, MPU-6500, BMP280)는 `cuas/hw/Hackathon.pdf`에 있습니다.


## 탐지 방식

`cuas/detect_only/pi_fb_detect.py`가 Pi에서 돌아가는 최종 코드입니다.

1. `rpicam-vid`로 640x480 YUV420, 30fps 영상을 받습니다. 별도 라이브러리(picamera2) 없이 OpenCV와 numpy만 씁니다.
2. 표시와 분석을 서로 다른 프로세스로 나눴습니다. 분석이 느려도 고글 영상은 끊기지 않습니다.
   - 표시 프로세스: 매 프레임 최근 박스를 그려 `/dev/fb0`(컴포지트 출력)에 씁니다.
   - 분석 프로세스: 가장 최근 프레임만 320x240으로 받아 초당 최대 10번 분석합니다.
3. 분석은 두 갈래입니다.
   - 멀리 있는 작은 표적: 배경 대비와 움직임으로 후보를 찾고 추적한 뒤(`motion_id.py`), 후보 조각을 외형 분류기로 판정합니다.
   - 가까이 있는 큰 드론: 화면에서 검은 큰 덩어리를 찾아 박스를 잡고, 그 조각을 외형 분류기로 다시 확인해 드론 확률 0.6 이상일 때만 남깁니다.
4. 시연에서는 드론만 표시합니다. 분류기는 새, 풍선, 오물풍선도 구분하지만 그 판정은 버립니다.
5. 결과는 고글 화면에 `DRONE DETECTED`와 박스로, 노트북에는 UDP로 보냅니다.

분석 프로세스가 죽으면 자동으로 다시 띄우고, 카메라나 화면 장치를 열 수 없으면 탐지 없이 영상만 내보내 고글이 꺼지지 않게 했습니다.


## 외형 분류 모델

- 구조: MobileNetV3-Small(ImageNet 사전학습) 미세조정, 입력 128x128, ONNX로 내보내 Pi에서는 OpenCV dnn으로 실행
- 클래스: balloon, bird, drone, none, trash
- 학습 환경: 노트북 GPU(RTX 4060), WSL, PyTorch
- 증강: 작게 줄였다 키우기(8~64px, 멀리 있는 표적 흉내), 흐림, 흑백(고글 화면), 색 변화, 클래스 균형 샘플링

데이터
- 새, 드론(일반): eashmanova/Drons_and_Birds_Detection
- 풍선: ImageNet balloon(mlnomad/imnet1k_balloon)
- 우리 드론: 팀 기체를 직접 찍은 사진 33장. 박스를 자동으로 잡은 뒤(`mydrone_boxes.py`) 틀린 4장은 손으로 고쳤고, 박스 주변을 여러 크기로 잘라 학습 조각을 만들었습니다(`build_mydrone.py`).
  같은 사진의 손, 벽, 사람, 배너 부분은 none 으로 넣어 손이나 벽을 드론으로 보지 않게 했습니다.

평가
- 사진 단위로 나눴습니다. 28장으로 학습하고 학습에 쓰지 않은 5장으로 시험했습니다(같은 사진의 조각이 학습과 시험에 섞이지 않게).
- 시험 결과: 드론 조각 60개 모두 드론으로 판정, 배경·새·풍선을 드론으로 잘못 본 경우 0건, 전체 균형 정확도 0.990
- 상세 수치: `cuas/appearance/out_v6_mydrone/report.json`

한계: 학습 사진은 휴대폰으로, 하루 같은 장소에서 찍었습니다. Pi 카메라로 따로 평가 세트를 만들지는 못했습니다.


## Pi 실측

- 고글 표시: 약 28~30fps
- 분석(박스 갱신): 약 3.5~5회/초 (가까운 드론 확인 단계를 넣은 뒤 기준)
- CPU 온도: 45~61도, 스로틀링 없음
- 탐지 확인: 실제 기체를 카메라 앞에 들었을 때 drone 0.82~0.93으로 연속 판정


## 관제 GUI

`cuas/detect_only/gui/detect_gui.py --pi`

- 왼쪽: Pi가 보내는 실시간 영상, 드론 박스와 확률, 화면 위 빨간 띠 "드론 탐지되었습니다"
- 오른쪽: 탐지 순간 캡처, 객체 설명과 대응 안내, 이벤트 로그 표
- 사람이 판정을 확인하거나 정정할 수 있고, Enter를 누르면 A.R.M.S. 시뮬에 대응 명령을 보냅니다.
- 신호가 2초 넘게 끊기면 "신호 없음"을 표시합니다.


## A.R.M.S. 시뮬레이션 (`sim/`)

- Gazebo Classic 11 + 직접 만든 플러그인(`sim/plugins/arms_bridge.cc`)
- 요격 드론은 온보드에서 알 수 있는 값만 씁니다: 카메라 검출 결과, FC 자세, 고도와 상승률. 표적의 실제 위치는 명중 판정에만 씁니다.
- 유도: 비례항법(PN)
- 관제 GUI에서 `intercept` 명령이 오면 리셋 후 자동 발진하고, 상태(발진, 추적, 명중)를 회신합니다. `track` 명령은 터뜨리지 않고 일정 거리에서 따라갑니다.
- `attach_sim.py`: 풍선 줄에 추적 태그를 거는 과정(접근, 대기, 접촉, 부착, 이탈) 시뮬
- 10개 상황(seed 1~10) 요격 명중 7/10


## 그 밖의 코드

- `trainer/`: FPV 조종 훈련기. Gazebo 트랙과 같은 세계를 Windows에서 실시간으로 그리고, 키보드·조종기·IMU 착용 컨트롤러로 방향을 조종해 장치별 학습 곡선을 비교합니다. `imu_nucode/`는 IMU 보드 펌웨어입니다.
- `imu_to_msp.py`: IMU 착용 컨트롤러 값을 Pi에서 Betaflight MSP RC override로 보냅니다. 교관 조종기의 AUX4 스위치가 켜져 있을 때만 동작합니다.
- `track3.py`: Pi에서 색 기반 표적 추적과 yaw 명령 생성
- `msp_link_test.py`: 카메라 없이 Pi와 FC 배선, 스위치 확인
- `cuas/` 바로 아래 파일들: 최종본 이전 버전(오물풍선 판정, 관제 로그 `ground_station.py`, 영상 평가 등)


## 실행 방법

### 준비
- 노트북: Python 3, `opencv-python`, `numpy`, `pillow`, `paramiko`
- Pi: Raspberry Pi OS, `python3`, OpenCV, `rpicam-apps`
- 시뮬: WSL Ubuntu 22.04, Gazebo Classic 11, `bash sim/setup_venv.sh`, `bash sim/build_plugins.sh`

Pi 접속 정보는 저장소에 넣지 않았습니다. 환경변수로 넘깁니다.

```
set CUAS_PI_USER=pi
set CUAS_PI_PASS=<비밀번호>
```

핫스팟 이름과 Pi 주소는 `cuas/detect_only/run_detect_only.py` 위쪽(`cuas-hotspot`, `172.20.10.13`)을 환경에 맞게 바꿉니다.

### Pi에 설치
SD카드로 설치(권장): 부트 파티션이 D: 일 때

```
python cuas/detect_only/write_sd.py D:/
```

다음 부팅 때 코드 복사, 서비스 등록 후 한 번 재부팅하고, 이후로는 켤 때마다 탐지가 자동으로 시작됩니다.

SSH로 바로 올리고 실행:

```
python cuas/detect_only/run_detect_only.py          # 업로드 + 탐지 시작 + GUI
python cuas/detect_only/run_detect_only.py --stop   # 원래 서비스로 되돌리기
```

### 관제 GUI
```
python cuas/detect_only/gui/detect_gui.py --pi
```

### 모델 다시 학습
```
python cuas/appearance/mydrone_boxes.py      # 사진에서 드론 박스 추정 + 확인용 이미지
python cuas/appearance/build_mydrone.py      # 학습 조각 생성
python3 cuas/appearance/train_cls.py --data <조각 폴더> --out <출력 폴더> --epochs 15
```

사진은 `cuas/appearance/mydrone_photos/`에 넣습니다(저장소에는 없음).

### 시뮬
```
cd sim
bash arms_demo.sh                                                    # 시연 화면
~/arms-venv/bin/python arms.py --nogui --episodes 20 --guidance PN   # 명중률 평가
```


## 폴더 구조

```
cuas/
  detect_only/        최종 시연본 (Pi 탐지, GUI, SD 설치, 실행 스크립트)
  appearance/         분류기 학습 코드, 학습된 모델(out_v6_mydrone), 샘플 사진 3장
  hw/                 하드웨어 구성도, 비행 컨트롤러 회로도
  gui/                이전 버전 GUI
sim/                  Gazebo 월드, 플러그인, 요격/부착 시뮬
trainer/              FPV 훈련기, IMU 펌웨어
docs/계획.pdf         시연 계획 초안
docs/계획최종.pdf     시연 계획 최종본 (문제 제기, 클래스별 대응, 화면 정의)
docs/presentation.pdf 발표 자료
```


## 데이터와 보안

- 비밀번호, 와이파이 키, 토큰은 넣지 않았습니다. Pi 접속 정보는 환경변수, 펌웨어 와이파이는 자리표시 값입니다.
- 원본 학습 사진, 학습 조각, 탐지 로그, 스냅샷, 영상은 넣지 않았습니다.
- 샘플 사진은 사람이 나오지 않은 기체 사진 3장만 크기를 줄이고 메타데이터를 지워서 넣었습니다.
