DICOM Viewer for Windows  (created by meş)
==========================================

INSTALL  (one time, about 5-10 minutes, no administrator rights needed)
  1. Right-click the ZIP file > "Extract All..." and open the extracted folder.
     (Do not run the installer from inside the ZIP.)
  2. Double-click "Install DICOM Viewer.bat".
     If Windows shows "Windows protected your PC": click "More info" > "Run anyway".
  3. Wait until it says "DICOM Viewer is installed". It downloads Python (only if
     you don't have it) and all packages - about 400 MB.

START
  Desktop shortcut "DICOM Viewer", or Start menu > type "DICOM".

UNINSTALL
  Settings > Apps > Installed apps > DICOM Viewer > Uninstall.

REQUIREMENTS
  Windows 10 or 11, 64-bit, internet connection during installation.
  3D view: a graphics driver with OpenGL 3.2 or newer (any GPU from the last ~10 years).
  If the 3D view cannot start, the 2D viewer still works; update the graphics driver.

WHERE THINGS ARE
  Program and private Python environment:  %LOCALAPPDATA%\DICOM Viewer
  Log file (send this if something goes wrong): %LOCALAPPDATA%\DICOM Viewer\dicom_viewer.log

UPDATING
  Put the new dicom_viewer.py next to the installer and run "Install DICOM Viewer.bat" again.

Research / education use only - not for diagnosis.
