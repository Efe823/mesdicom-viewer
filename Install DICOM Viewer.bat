@echo off
title DICOM Viewer - installer
echo.
echo  DICOM Viewer installer
echo  ----------------------
echo  Installs Python (if needed), all required packages and the shortcuts.
echo  No administrator rights needed. Internet connection required (about 400 MB).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
