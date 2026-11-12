# DICOM Viewer - Windows installer (per-user, no admin rights needed)
#  1. finds a 64-bit Python 3.10-3.13, or installs Python 3.12.10 from python.org
#  2. copies the program to %LOCALAPPDATA%\DICOM Viewer
#  3. creates a private virtual environment and installs every package
#  4. creates Start menu + Desktop shortcuts and an entry in Settings > Apps
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"          # much faster downloads in Windows PowerShell 5.1

$AppName    = "DICOM Viewer"
$Here       = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Join-Path $env:LOCALAPPDATA $AppName
$PyVersion  = "3.12.10"
$PyUrl      = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"
$Packages   = @("pydicom", "PyQt6", "numpy", "scipy", "scikit-image", "vtk",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Finish($code) { Write-Host ""; Read-Host "Press Enter to close this window" | Out-Null; exit $code }
function Fail($msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; Finish 1 }

trap { Write-Host ""; Write-Host "ERROR: $_" -ForegroundColor Red; Finish 1 }

if (-not [Environment]::Is64BitOperatingSystem) { Fail "64-bit Windows is required." }
if (-not (Test-Path (Join-Path $Here "dicom_viewer.py"))) {
    Fail "dicom_viewer.py was not found next to the installer. Extract the whole ZIP first, then run the installer from the extracted folder."
}
if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    Write-Host "Windows on ARM detected: the 64-bit (x64) Python is used through emulation so all packages are available." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 1. Python
function Find-Python {
    $ErrorActionPreference = "Continue"
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.13", "3.10")) { $candidates += ,@("py", "-$v") }
    }
    $own = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $own) { $candidates += ,@($own) }
    $candidates += ,@("python")
    $check = "import sys,struct; ok=(3,10)<=sys.version_info[:2]<=(3,13) and struct.calcsize('P')==8; print(sys.executable if ok else '')"
    foreach ($c in $candidates) {
        try {
            $exe = $c[0]
            $pyArgs = @()
            if ($c.Count -gt 1) { $pyArgs = $c[1..($c.Count - 1)] }
            $out = & $exe @pyArgs -c $check 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $path = ($out | Select-Object -Last 1).ToString().Trim()
                if ($path -and (Test-Path $path) -and ($path -notlike "*WindowsApps*")) { return $path }
            }
        } catch { }
    }
    return $null
}

Step "Looking for Python 3.10 - 3.13 (64-bit)"
$Python = Find-Python
if ($Python) {
    Write-Host "Found: $Python"
} else {
    Step "Downloading Python $PyVersion from python.org"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $setup = Join-Path $env:TEMP "python-$PyVersion-amd64.exe"
    Invoke-WebRequest -Uri $PyUrl -OutFile $setup -UseBasicParsing
    Step "Installing Python $PyVersion for this user (takes about a minute)"
    $p = Start-Process -FilePath $setup -Wait -PassThru -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Shortcuts=0", "AssociateFiles=0")
    if ($p.ExitCode -ne 0) { Fail "The Python installer failed (exit code $($p.ExitCode))." }
    Remove-Item $setup -Force -ErrorAction SilentlyContinue
    $Python = Find-Python
    if (-not $Python) { Fail "Python was installed but could not be found. Please run the installer again." }
    Write-Host "Installed: $Python"
}

# ---------------------------------------------------------------- 2. program files
Step "Copying the program to $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item (Join-Path $Here "dicom_viewer.py") $InstallDir -Force
Copy-Item (Join-Path $Here "uninstall_windows.ps1") $InstallDir -Force
$Script = Join-Path $InstallDir "dicom_viewer.py"

# ---------------------------------------------------------------- 3. environment + packages
$Venv = Join-Path $InstallDir ".venv"
$VPy  = Join-Path $Venv "Scripts\python.exe"
$VPyw = Join-Path $Venv "Scripts\pythonw.exe"
if (-not (Test-Path $VPy)) {
    Step "Creating a private Python environment"
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the Python environment." }
}
Step "Installing packages - about 400 MB, this can take several minutes"
& $VPy -m pip install --upgrade pip --disable-pip-version-check
& $VPy -m pip install --upgrade --disable-pip-version-check @Packages
if ($LASTEXITCODE -ne 0) {
    Fail "Package installation failed. Check the internet connection and run the installer again."
}

# ---------------------------------------------------------------- 4. shortcuts + Settings > Apps
Step "Creating Start menu and Desktop shortcuts"
& $VPy $Script --install-launcher
if ($LASTEXITCODE -ne 0) { Fail "Could not create the shortcuts." }

$Key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DICOMViewer"
New-Item -Path $Key -Force | Out-Null
$sizeKB = [int]((Get-ChildItem $InstallDir -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object Length -Sum).Sum / 1KB)
$values = @{
    DisplayName     = $AppName
    DisplayVersion  = "1.0"
    Publisher       = "created by meş"
    InstallLocation = $InstallDir
    DisplayIcon     = (Join-Path $InstallDir "$AppName.ico")
    UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$InstallDir\uninstall_windows.ps1`""
}
foreach ($k in $values.Keys) { New-ItemProperty -Path $Key -Name $k -Value $values[$k] -PropertyType String -Force | Out-Null }
foreach ($k in @("NoModify", "NoRepair")) { New-ItemProperty -Path $Key -Name $k -Value 1 -PropertyType DWord -Force | Out-Null }
New-ItemProperty -Path $Key -Name "EstimatedSize" -Value $sizeKB -PropertyType DWord -Force | Out-Null

Step "Checking the installation"
& $VPy $Script --selftest

Write-Host ""
Write-Host "DICOM Viewer is installed." -ForegroundColor Green
Write-Host "Start it from the Desktop shortcut or the Start menu (type 'DICOM')."
Write-Host "Uninstall: Settings > Apps > Installed apps > DICOM Viewer."
$answer = Read-Host "Start DICOM Viewer now? (Y/n)"
if ($answer -notmatch "^[nN]") { Start-Process -FilePath $VPyw -ArgumentList "`"$Script`"" -WorkingDirectory $InstallDir }
exit 0
