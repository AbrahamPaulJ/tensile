# Sets up the local text-to-speech used for the demo video's narration: Kokoro (Apache-2.0), run on the
# CPU through kokoro-onnx, in a throwaway virtual environment under the user's cache folder.
# usage: powershell -File demo/video/setup_tts.ps1
$ErrorActionPreference = 'Stop'
$cache = Join-Path $env:LOCALAPPDATA 'tensile-tts'
New-Item -ItemType Directory -Force $cache | Out-Null
if (-not (Test-Path "$cache\venv")) { python -m venv "$cache\venv" }
& "$cache\venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check kokoro-onnx soundfile
foreach ($file in 'kokoro-v1.0.onnx', 'voices-v1.0.bin') {
  if (-not (Test-Path "$cache\$file")) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$file" -OutFile "$cache\$file"
  }
  "{0} {1} bytes" -f $file, (Get-Item "$cache\$file").Length
}
& "$cache\venv\Scripts\python.exe" -c "import kokoro_onnx; print('kokoro_onnx ready')"
