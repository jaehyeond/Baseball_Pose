# MMPose 3D Triangulation Runner Script
# 사용법: .\run_triangulation.ps1

param(
    [int]$StartFrame = 0,
    [int]$EndFrame = 50,
    [double]$ConfThreshold = 0.4,
    [double]$ReprojThreshold = 8.0
)

Write-Host "🚀 MMPose 3D Triangulation Pipeline" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# 1. 가상환경 활성화 확인
Write-Host "1. 가상환경 활성화 중..." -ForegroundColor Yellow
if (Test-Path "mmpose_env\Scripts\Activate.ps1") {
    .\mmpose_env\Scripts\Activate.ps1
    Write-Host "   ✅ 가상환경 활성화됨" -ForegroundColor Green
} else {
    Write-Host "   ❌ 가상환경을 찾을 수 없습니다!" -ForegroundColor Red
    exit 1
}

# 2. Python 경로 확인
Write-Host "2. Python 경로 확인..." -ForegroundColor Yellow
$pythonPath = (Get-Command python).Source
Write-Host "   Python Path: $pythonPath" -ForegroundColor Cyan

if ($pythonPath -notlike "*mmpose_env*") {
    Write-Host "   ⚠️  가상환경이 제대로 활성화되지 않았을 수 있습니다." -ForegroundColor Yellow
}

# 3. 필수 파일 존재 확인
Write-Host "3. 필수 파일 확인..." -ForegroundColor Yellow
$requiredFiles = @(
    "calibration\azure_kinect_calibration.json",
    "results_2d_camera1",
    "results_2d_camera2",
    "twocam_test\robust_triangulation.py",
    "demo\robust_visualization_final.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file 없음" -ForegroundColor Red
        exit 1
    }
}

# 4. 삼각측량 실행
Write-Host "4. 강력한 삼각측량 실행..." -ForegroundColor Yellow
Write-Host "   프레임: $StartFrame ~ $EndFrame" -ForegroundColor Cyan
Write-Host "   신뢰도 임계값: $ConfThreshold" -ForegroundColor Cyan
Write-Host "   재투영 오차 임계값: $ReprojThreshold px" -ForegroundColor Cyan

$triangulationCmd = "python twocam_test/robust_triangulation.py --calib-file calibration/azure_kinect_calibration.json --cam1-poses results_2d_camera1 --cam2-poses results_2d_camera2 --output-file output_3d/robust_3d_results.json --conf-threshold $ConfThreshold --reproj-threshold $ReprojThreshold --start-frame $StartFrame --end-frame $EndFrame"

Write-Host "   명령어: $triangulationCmd" -ForegroundColor Gray
Invoke-Expression $triangulationCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ 삼각측량 완료" -ForegroundColor Green
} else {
    Write-Host "   ❌ 삼각측량 실패" -ForegroundColor Red
    exit 1
}

# 5. 시각화 실행
Write-Host "5. 3D 시각화 실행..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "vis3d_$timestamp"

$visualizationCmd = "python demo/robust_visualization_final.py --input output_3d/robust_3d_results.json --out-dir $outputDir --start $StartFrame --end $EndFrame"

Write-Host "   명령어: $visualizationCmd" -ForegroundColor Gray
Invoke-Expression $visualizationCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ 시각화 완료" -ForegroundColor Green
    Write-Host "   📁 결과 위치: $outputDir" -ForegroundColor Cyan
} else {
    Write-Host "   ❌ 시각화 실패" -ForegroundColor Red
    exit 1
}

# 6. 완료 메시지
Write-Host "🎉 전체 파이프라인 완료!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "📊 결과 요약:" -ForegroundColor White
Write-Host "   - 처리 프레임: $($EndFrame - $StartFrame + 1)개" -ForegroundColor White
Write-Host "   - 3D 데이터: output_3d/robust_3d_results.json" -ForegroundColor White
Write-Host "   - 시각화: $outputDir/" -ForegroundColor White
Write-Host ""
Write-Host "🔍 결과 확인 방법:" -ForegroundColor Cyan
Write-Host "   - 3D 이미지: $outputDir/robust_frame_*.png" -ForegroundColor White
Write-Host "   - JSON 데이터: output_3d/robust_3d_results.json" -ForegroundColor White