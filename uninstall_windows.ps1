# DICOM Viewer - uninstaller (also used by Settings > Apps)
$ErrorActionPreference = "Continue"
$AppName    = "DICOM Viewer"
$InstallDir = Join-Path $env:LOCALAPPDATA $AppName
$VPy        = Join-Path $InstallDir ".venv\Scripts\python.exe"
$Script     = Join-Path $InstallDir "dicom_viewer.py"

Write-Host "This removes DICOM Viewer, its shortcuts and its private Python environment."
Write-Host "(Python itself and your DICOM files are not touched.)"
$answer = Read-Host "Uninstall DICOM Viewer? (y/N)"
if ($answer -notmatch "^[yY]") { exit 0 }

# close running copies
Get-Process python, pythonw -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase) } |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500

if ((Test-Path $VPy) -and (Test-Path $Script)) { & $VPy $Script --uninstall-launcher }
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    $lnk = Join-Path $dir "$AppName.lnk"
    if (Test-Path $lnk) { Remove-Item $lnk -Force }
}
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\DICOMViewer" -Recurse -Force -ErrorAction SilentlyContinue
Set-Location $env:TEMP
Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $InstallDir) {
    Write-Host "Some files are still in use. Close DICOM Viewer and delete this folder manually:" -ForegroundColor Yellow
    Write-Host "  $InstallDir"
} else {
    Write-Host "DICOM Viewer was removed." -ForegroundColor Green
}
Read-Host "Press Enter to close" | Out-Null
