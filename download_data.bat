@echo off
echo ========================================================
echo       ASVspoof: Downloading Datasets
echo ========================================================
echo.

:: Check for curl
where curl >nul 2>&1
if %errorlevel% equ 0 (
    set USE_CURL=true
    echo [INFO] curl is available. Will use curl with resume support.
) else (
    set USE_CURL=false
    echo [WARNING] curl is not found! Will fallback to PowerShell Invoke-WebRequest.
    echo [WARNING] Note: PowerShell download does not support resuming interrupted downloads.
)
echo.

:: Ensure data directories exist
if not exist data mkdir data
if not exist data\2019 mkdir data\2019
if not exist data\2021 mkdir data\2021

:: 1. Download ASVspoof 2019 LA Dataset (7.6 GB)
if not exist data\2019\LA.zip (
    echo [INFO] Downloading ASVspoof 2019 LA Dataset (7.6 GB)...
    if "%USE_CURL%"=="true" (
        curl -L -C - -o data\2019\LA.zip https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip' -OutFile 'data\2019\LA.zip'"
    )
) else (
    echo [INFO] ASVspoof 2019 LA Dataset (LA.zip) already exists. Skipping download.
)
echo.

:: 2. Download ASVspoof 2021 LA Eval Dataset (7.7 GB)
if not exist data\2021\ASVspoof2021_LA_eval.tar.gz (
    echo [INFO] Downloading ASVspoof 2021 LA Eval Dataset (7.7 GB)...
    if "%USE_CURL%"=="true" (
        curl -L -C - -o data\2021\ASVspoof2021_LA_eval.tar.gz "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1' -OutFile 'data\2021\ASVspoof2021_LA_eval.tar.gz'"
    )
) else (
    echo [INFO] ASVspoof 2021 LA Eval Dataset (ASVspoof2021_LA_eval.tar.gz) already exists. Skipping download.
)
echo.

:: 3. Download ASVspoof 2021 Keys (21 MB)
if not exist data\2021\LA-keys-full.tar.gz (
    echo [INFO] Downloading ASVspoof 2021 Keys (21 MB)...
    if "%USE_CURL%"=="true" (
        curl -L -C - -o data\2021\LA-keys-full.tar.gz https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
    ) else (
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz' -OutFile 'data\2021\LA-keys-full.tar.gz'"
    )
) else (
    echo [INFO] ASVspoof 2021 Keys (LA-keys-full.tar.gz) already exists. Skipping download.
)
echo.

echo ========================================================
echo [SUCCESS] Datasets download completed!
echo.
echo You can now run training/evaluation scripts.
echo ========================================================
pause
