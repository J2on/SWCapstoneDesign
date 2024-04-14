@echo off
set DIR=%~dp0
rd /s /q "%DIR%Intermediate"
rd /s /q "%DIR%DerivedDataCache"
rd /s /q "%DIR%Binaries"
rd /s /q "%DIR%Saved"
del /f /q "MediaPipe4U-Demo.sln"
@echo on
@echo project clear succeed !! Recompole it use VisualStudio or Rider.
pause