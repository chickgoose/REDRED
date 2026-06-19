# REDRED 팀 진행상황 매뉴얼

## 서버 접속 방법

### 1. TurboX 웹 접속
- 브라우저에서 TurboX 주소 접속 (학교 VPN 필요할 수 있음)
- 로그인 후 xterm 터미널 열기
- xterm을 열면 **자동으로 Singularity 컨테이너 안에서 시작됨**

### 2. SSH 일반 접속 시 Singularity 수동 진입 (중요!)
TurboX 없이 SSH로 접속하면 Singularity 컨테이너 밖에 있어서 GPU 사용 불가.
아래 명령어로 수동으로 컨테이너 진입:

```bash
# ssai_agpu 큐 (MIG GPU, 대기 있을 수 있음)
qsub -q ssai_agpu -l select=1:ncpus=1:mem=36g:ngpus=1:Qlist=mig_agpu:container_engine=singularity \
  -v "CONTAINER_IMAGE=147.46.121.38:5000/ubuntu:18.04-gpu,PBS_CONTAINER_ARGS=--no-https" \
  -I -- /tools/scripts/pbs_bash.sh

# ssu_a6gpu 큐 (A6000 GPU, 대기 적음 - 추천)
qsub -q ssu_a6gpu -l select=1:ncpus=6:mem=128g:ngpus=1:Qlist=a6000:container_engine=singularity \
  -v "CONTAINER_IMAGE=147.46.121.38:5000/ubuntu:18.04-gpu,PBS_CONTAINER_ARGS=--no-https" \
  -I -- /tools/scripts/pbs_bash.sh
```

- `qsub: waiting for job XXXXX.ECE-util1 to start` 메시지 뜨면 GPU 할당 대기 중 → 기다리면 됨
- 큐 상태 확인: `qstat -q ssu_a6gpu`
- 접속되면 프롬프트가 `(yolov7) Singularity>` 로 바뀜

### 3. conda 환경 활성화
```bash
conda activate ~/envs/yolov7
```

---

## 서버 디렉토리 구조

```
~/
├── Dataset/
│   ├── yolov7_custom.pt          # 대회 제공 원본 가중치 (mAP@0.5=98.1%)
│   ├── names.txt                 # 60개 클래스 이름
│   ├── 1.competition_trainset/   # 학습 데이터 (20,436장)
│   ├── 3.Background_Images/      # 배경 이미지 (2,915장)
│   ├── 3.background_substracted_white/  # 세그멘테이션 이미지 (클래스별 폴더)
│   ├── 4.TestVideo_Sample/       # 테스트 영상 (cam0~cam4)
│   └── augmented/                # Cut&Paste 증강 이미지 (5,000장)
├── yolov7/                       # YOLOv7 코드
│   ├── data/
│   │   ├── custom.yaml           # 학습 설정 (nc=60)
│   │   ├── train.txt             # 학습 이미지 경로 (23,392줄)
│   │   └── val.txt               # 검증 이미지 경로 (2,043줄)
│   └── runs/train/retrain_aug/   # 파인튜닝 결과 (mAP 오히려 낮아짐)
├── REDRED/                       # 이 레포
└── envs/yolov7/                  # conda 환경 (Python 3.8, torch 1.12.1+cu113)
```

---

## 완료된 작업

### Step 1: 파이프라인 구축 ✅
- 5개 카메라 영상 → YOLOv7 추론 → 멀티뷰 퓨전 → 이벤트 감지 → CSV 출력
- RTF 0.751 달성 (기준 1.0 이하)
- 192개 이벤트 감지
- 최종 제출 파일: `~/REDRED/output/submission_skip2.csv`

### Step 2: RTF 최적화 ✅
- `grab()` 사용으로 skip 프레임 디코딩 없이 위치만 이동
- 배치 추론 (5개 카메라 프레임 → 단일 GPU forward pass)
- skip=2로 실질적인 RTF 절감

### Step 3: Cut&Paste 증강 ✅
- `augment/cut_paste_aug.py`: Flip + Perspective Warp + Blur/Noise
- 5,000장 생성 완료 (`~/Dataset/augmented/`)
- 증강 명령: `bash ~/REDRED/run_augment.sh 5000`

### Step 4: 파인튜닝 시도 (결과 미채택)
- 원본 가중치 기반 30 epoch 추가 학습
- mAP@0.5: 0.9904 (원본) → 0.9840 (파인튜닝) — 오히려 낮아짐
- 원인: Cut&Paste 증강과 실제 데이터 분포 차이
- **현재는 원본 가중치(`yolov7_custom.pt`) 사용 중**

### Step 5: 가격 데이터 ✅
- `data/prices.csv`: 60개 클래스 전체 가격 입력 완료

---

## 파이프라인 실행 방법

```bash
# Singularity 진입 후
conda activate ~/envs/yolov7
cd ~/REDRED && git pull

# skip=2로 실행 (RTF 최적화, 추천)
bash run_test.sh 2

# skip=1로 실행 (정확도 최대)
bash run_test.sh 1
```

결과 파일: `~/REDRED/output/submission_skip{N}.csv`

---

## 재학습 방법 (필요 시)

```bash
# 증강 데이터 새로 생성
bash ~/REDRED/run_augment.sh 5000

# train.txt 재구성
grep -v "augmented" ~/yolov7/data/train.txt > /tmp/train_clean.txt
mv /tmp/train_clean.txt ~/yolov7/data/train.txt
find ~/Dataset/augmented/images -name "*.jpg" >> ~/yolov7/data/train.txt

# 재학습 (screen으로 세션 유지)
screen -S train
cd ~/yolov7 && PYTHONPATH=~/yolov7 python train.py \
    --weights ~/Dataset/yolov7_custom.pt \
    --data data/custom.yaml \
    --epochs 30 \
    --batch-size 16 \
    --img 640 \
    --device 0 \
    --name retrain_v2 \
    --exist-ok

# 학습 중 세션 이탈: Ctrl+A, D
# 재접속: screen -r train
```

---

## Git 레포

- **chickgoose/REDRED**: https://github.com/chickgoose/REDRED (박준영)
- **GangHeeJo/REDRED**: https://github.com/GangHeeJo/REDRED (강희조)

서버에서 pull:
```bash
cd ~/REDRED && git pull
```

---

## 현재 상태 요약

| 항목 | 상태 |
|------|------|
| 파이프라인 | ✅ 완료 |
| RTF | ✅ 0.751 (기준 1.0 이하) |
| 이벤트 감지 | ✅ 192 events |
| 가격 데이터 | ✅ 입력 완료 |
| 제출 CSV | ✅ submission_skip2.csv |
| 재학습 | ❌ mAP 하락으로 미채택 |
| 발표 자료 | ⬜ 미완료 |
