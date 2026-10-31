@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
if not exist "%ROOT%\runs" mkdir "%ROOT%\runs"
echo Hybrid stop requested at %DATE% %TIME%>"%ROOT%\runs\HYBRID_STOP"
echo Requested cooperative stop. The supervisor will stop both CPU and quantum workers at safe boundaries.
