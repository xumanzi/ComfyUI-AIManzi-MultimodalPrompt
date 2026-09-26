$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$pluginRoot = $PSScriptRoot
$toolsRoot = Join-Path $pluginRoot "tools"
$installRoot = Join-Path $toolsRoot "ffmpeg"
$settingsPath = Join-Path $pluginRoot "config\settings.json"
$downloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$archivePath = Join-Path ([IO.Path]::GetTempPath()) ("aimanzi-ffmpeg-" + [Guid]::NewGuid().ToString("N") + ".zip")
$extractPath = Join-Path ([IO.Path]::GetTempPath()) ("aimanzi-ffmpeg-" + [Guid]::NewGuid().ToString("N"))

Write-Host "[AIManzi] Installing FFmpeg for video input..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $toolsRoot -Force | Out-Null

try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -UseBasicParsing
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force
    $ffmpeg = Get-ChildItem -LiteralPath $extractPath -Filter "ffmpeg.exe" -File -Recurse | Select-Object -First 1
    if (-not $ffmpeg) { throw "Downloaded package does not contain ffmpeg.exe." }

    if (Test-Path -LiteralPath $installRoot) {
        Remove-Item -LiteralPath $installRoot -Recurse -Force
    }
    $installedBin = Join-Path $installRoot "bin"
    New-Item -ItemType Directory -Path $installedBin -Force | Out-Null
    Copy-Item -Path (Join-Path $ffmpeg.Directory.FullName "*") -Destination $installedBin -Recurse -Force
    $installedFfmpeg = Join-Path $installedBin "ffmpeg.exe"
    $installedFfprobe = Join-Path $installedBin "ffprobe.exe"
    if (-not (Test-Path -LiteralPath $installedFfmpeg) -or -not (Test-Path -LiteralPath $installedFfprobe)) {
        throw "FFmpeg installation verification failed."
    }

    if (-not (Test-Path -LiteralPath $settingsPath)) {
        throw "Plugin settings file was not found: $settingsPath"
    }
    $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $settings.ffmpeg = $installedFfmpeg
    $settingsJson = $settings | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText($settingsPath, $settingsJson + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))

    Write-Host "[AIManzi] FFmpeg installed and configured:" -ForegroundColor Green
    Write-Host "  $installedFfmpeg"
    Write-Host "[AIManzi] Restart ComfyUI before using VIDEO input." -ForegroundColor Yellow
} finally {
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
    if (Test-Path -LiteralPath $extractPath) { Remove-Item -LiteralPath $extractPath -Recurse -Force }
}
