# MMPose 3D Triangulation Setup Guide

## 🚀 Quick Start

### 1. 가상환경 활성화
```powershell
# PowerShell에서 실행
.\mmpose_env\Scripts\Activate.ps1
```

### 2. Python 경로 확인
```powershell
python --version
# 출력 예시: Python 3.9.x (가상환경 경로가 표시되어야 함)
```

### 3. 패키지 확인
```powershell
pip list | findstr mmpose
pip list | findstr numpy
```

## 📋 주요 명령어들

### 강력한 삼각측량 실행
```powershell
# 가상환경 활성화 (매번 필수)
.\mmpose_env\Scripts\Activate.ps1

# 삼각측량 실행 (20프레임)
python twocam_test/robust_triangulation.py --calib-file calibration/azure_kinect_calibration.json --cam1-poses results_2d_camera1 --cam2-poses results_2d_camera2 --output-file output_3d/robust_3d_results.json --conf-threshold 0.4 --reproj-threshold 8.0 --start-frame 0 --end-frame 20
```

### 3D 시각화 실행
```powershell
# 강력한 시각화
python demo/robust_visualization_final.py --input output_3d/robust_3d_results.json --out-dir vis3d_ROBUST_FINAL --start 0 --end 20
```

### 전체 파이프라인 (한번에)
```powershell
# 1. 가상환경 활성화
.\mmpose_env\Scripts\Activate.ps1

# 2. 삼각측량
python twocam_test/robust_triangulation.py --calib-file calibration/azure_kinect_calibration.json --cam1-poses results_2d_camera1 --cam2-poses results_2d_camera2 --output-file output_3d/robust_3d_results.json --conf-threshold 0.4 --reproj-threshold 8.0 --start-frame 0 --end-frame 50

# 3. 시각화
python demo/robust_visualization_final.py --input output_3d/robust_3d_results.json --out-dir vis3d_COMPLETE --start 0 --end 50
```

## 🛠️ 트러블슈팅

### PowerShell 실행 정책 오류
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 가상환경이 없는 경우
```powershell
python -m venv mmpose_env
.\mmpose_env\Scripts\Activate.ps1
pip install numpy matplotlib scipy opencv-python
```

### Python 경로 문제
```powershell
# 가상환경 활성화 확인
where python
# 출력: E:\mmpose\mmpose_env\Scripts\python.exe 이어야 함
```

## 📁 주요 파일들

- `calibration/azure_kinect_calibration.json` - 카메라 캘리브레이션
- `results_2d_camera1/` - Camera1의 2D 포즈 결과
- `results_2d_camera2/` - Camera2의 2D 포즈 결과
- `output_3d/robust_3d_results.json` - 3D 삼각측량 결과
- `vis3d_COMPLETE/` - 최종 3D 시각화 결과

## 🎯 성능 파라미터

### 삼각측량 파라미터
- `--conf-threshold 0.4`: 2D keypoint 신뢰도 임계값
- `--reproj-threshold 8.0`: 재투영 오차 임계값 (픽셀)
- `--start-frame 0 --end-frame 50`: 처리할 프레임 범위

### 품질 설정
- **높은 정확도**: `--conf-threshold 0.6 --reproj-threshold 5.0`
- **균형**: `--conf-threshold 0.4 --reproj-threshold 8.0` (권장)
- **빠른 처리**: `--conf-threshold 0.3 --reproj-threshold 12.0`

## ⚡ 협업 시 주의사항

1. **항상 가상환경 활성화**: `.\mmpose_env\Scripts\Activate.ps1`
2. **경로 확인**: `where python`으로 올바른 Python 사용 중인지 확인
3. **절대경로 사용 금지**: 모든 명령어는 상대경로로 작성
4. **결과 백업**: `vis3d_*` 폴더들은 크니까 필요한 것만 보관