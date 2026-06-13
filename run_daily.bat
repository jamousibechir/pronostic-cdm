@echo off
REM ============================================================
REM  Mise a jour quotidienne du pronostic CdM 2026
REM  Lance par le Planificateur de taches Windows (tous les jours 8h)
REM  Journal : data\daily.log
REM ============================================================
cd /d "C:\Users\bachb\Desktop\pronostic-cdm"
set PYTHONIOENCODING=utf-8

REM Interpreteur Python (chemin resolu ; repli sur le PATH)
set "PYEXE=C:\Users\bachb\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo. >> data\daily.log
echo ====================================================== >> data\daily.log
echo Run: %DATE% %TIME% >> data\daily.log
"%PYEXE%" update_daily.py >> data\daily.log 2>&1
echo Exit code: %ERRORLEVEL% >> data\daily.log
