# Auto-restart trainer on crash. Resumes from latest checkpoint if present.
# Run from world_model/.
#
# Usage:
#   .\scripts\go_forever.ps1

$ErrorActionPreference = "Continue"
$RunsDir = "runs/phase1a_wm"
New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null
$LogPath = "$RunsDir/train.log"

while ($true) {
    $latest = Join-Path $RunsDir "latest.pt"

    if (Test-Path $latest) {
        Write-Host "[go_forever] Resuming from $latest"
        & python -u scripts/train.py --resume $latest 2>&1 | Tee-Object -FilePath $LogPath -Append
    } else {
        Write-Host "[go_forever] Fresh start"
        & python -u scripts/train.py 2>&1 | Tee-Object -FilePath $LogPath -Append
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[go_forever] Training completed cleanly. Exiting."
        break
    }

    Write-Host "[go_forever] Exit code $LASTEXITCODE. Restarting in 10s..."
    Start-Sleep -Seconds 10
}
