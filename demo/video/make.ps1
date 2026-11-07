# Builds tensile-demo.mp4 end to end. Run from anywhere:
#   powershell -File demo/video/make.ps1 -Skill <path to a checkout of kajisho5/ffmpeg-skill>
# Needs: setup_tts.ps1 done once, Chrome, Node with demo/ installed, ffmpeg with libass, Python with NumPy.
param([Parameter(Mandatory = $true)][string]$Skill)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
Push-Location (Resolve-Path "$here\..\..")
try {
  & "$env:LOCALAPPDATA\tensile-tts\venv\Scripts\python.exe" demo/video/tts.py
  if ($LASTEXITCODE) { throw 'tts.py failed' }
  node demo/video/record.mjs
  if ($LASTEXITCODE) { throw 'record.mjs failed' }
  python demo/video/music.py
  if ($LASTEXITCODE) { throw 'music.py failed' }
  python demo/video/assemble.py $Skill
  if ($LASTEXITCODE) { throw 'assemble.py failed' }
} finally { Pop-Location }
