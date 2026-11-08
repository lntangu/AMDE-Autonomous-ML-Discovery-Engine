@echo off
setlocal
cd /d "%~dp0"
title AMDE 5.0 Super Learner Domination Benchmark (10000/100)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_domination_benchmark.ps1
pause
