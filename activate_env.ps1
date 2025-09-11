# MMPose 가상환경 활성화 스크립트
# 사용법: .\activate_env.ps1

Write-Host "🔧 MMPose 가상환경 활성화" -ForegroundColor Green

# 가상환경 존재 확인
if (Test-Path "mmpose_env\Scripts\Activate.ps1") {
    Write-Host "가상환경을 활성화합니다..." -ForegroundColor Yellow
    .\mmpose_env\Scripts\Activate.ps1
    
    # Python 경로 확인
    $pythonPath = (Get-Command python).Source
    Write-Host "✅ 활성화 완료" -ForegroundColor Green
    Write-Host "Python 경로: $pythonPath" -ForegroundColor Cyan
    
    # 패키지 확인
    Write-Host "`n📦 설치된 주요 패키지:" -ForegroundColor Yellow
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy; print(f'NumPy: {numpy.__version__}')
except: print('NumPy: 미설치')
try:
    import cv2; print(f'OpenCV: {cv2.__version__}')
except: print('OpenCV: 미설치')
try:
    import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')  
except: print('Matplotlib: 미설치')
"
    
    Write-Host "`n🎯 이제 다음 명령어들을 사용할 수 있습니다:" -ForegroundColor Cyan
    Write-Host "   python twocam_test/robust_triangulation.py --help" -ForegroundColor White
    Write-Host "   python demo/robust_visualization_final.py --help" -ForegroundColor White
    Write-Host "   .\run_triangulation.ps1" -ForegroundColor White
    
} else {
    Write-Host "❌ 가상환경을 찾을 수 없습니다!" -ForegroundColor Red
    Write-Host "다음 명령어로 가상환경을 생성하세요:" -ForegroundColor Yellow
    Write-Host "   python -m venv mmpose_env" -ForegroundColor White
    exit 1
}