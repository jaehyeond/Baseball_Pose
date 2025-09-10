@echo off
setlocal enabledelayedexpansion

echo 파일명 순차 수정 시작...

REM 00_test 폴더 처리
echo 00_test 폴더 처리 중...
cd /d "D:\mmpose\frames\00_test"

REM 역순으로 파일명 변경 (충돌 방지)
set /a counter=120
for /f "tokens=*" %%f in ('dir /b /o:n 00*.png ^| findstr /r "00[1-9][4-9][0-9]\.png"') do (
    set filename=%%f
    set old_num=!filename:~2,3!
    if !old_num! GEQ 141 (
        set new_name=00!counter!.png
        if !counter! LSS 100 (
            set new_name=000!counter!.png
        )
        echo Renaming !filename! to !new_name!
        ren "!filename!" "!new_name!"
        set /a counter+=1
    )
)

REM 01_test 폴더 처리
echo 01_test 폴더 처리 중...
cd /d "D:\mmpose\frames\01_test"

REM 역순으로 파일명 변경 (충돌 방지)
set /a counter=120
for /f "tokens=*" %%f in ('dir /b /o:n 00*.png ^| findstr /r "00[1-9][4-9][0-9]\.png"') do (
    set filename=%%f
    set old_num=!filename:~2,3!
    if !old_num! GEQ 141 (
        set new_name=00!counter!.png
        if !counter! LSS 100 (
            set new_name=000!counter!.png
        )
        echo Renaming !filename! to !new_name!
        ren "!filename!" "!new_name!"
        set /a counter+=1
    )
)

echo 파일명 수정 완료!
pause