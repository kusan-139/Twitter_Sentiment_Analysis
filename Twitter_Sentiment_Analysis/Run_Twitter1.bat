@echo off
:: Sets the title of the command prompt window
title Twitter Sentiment Analysis Tool

:: Clears the screen
cls

echo ========================================
echo  Twitter Sentiment Analysis Tool
echo ========================================
echo.
echo 🐍 Launching Python script in a new CMD window...
echo Please wait, this may take a moment on the first run.
echo.

:: Go to the directory where this batch file is located
cd /d "%~dp0"

:: Open a NEW command prompt window and run the Python script inside it
start cmd /k "python Twitter1.py & echo. & echo ======================================== & echo  Analysis Complete. & echo ======================================== & pause"

:: Exit the current batch window
exit
