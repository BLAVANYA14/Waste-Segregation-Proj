# Prepare dataset: copy dataset-resized contents to data\raw\trashnet and cleanup temp folders
try {
    $found = Get-ChildItem -Path .\data -Recurse -Directory -Filter 'dataset-resized' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    if (-not $found) {
        Write-Host 'No dataset-resized directories found'
        exit 0
    }
    $src = $null
    foreach ($d in $found) {
        if (Test-Path (Join-Path $d 'cardboard')) { $src = $d; break }
    }
    if (-not $src) { $src = $found[0] }
    Write-Host "Using source: $src"
    $dest = Join-Path (Get-Location) 'data\raw\trashnet'
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Write-Host "Copying dataset contents to $dest"
    Copy-Item -Path (Join-Path $src '*') -Destination $dest -Recurse -Force
    Write-Host 'Removing __MACOSX dirs inside destination (if any)'
    Get-ChildItem -Path $dest -Recurse -Directory -Force | Where-Object { $_.Name -eq '__MACOSX' } | ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    $toRemove = @('.\data\tmp_extracted','.\data\trashnet') | Where-Object { Test-Path $_ }
    if ($toRemove) {
        Write-Host 'Removing temporary folders:'
        foreach ($r in $toRemove) { Write-Host (' - ' + $r); Remove-Item -Recurse -Force $r }
    } else { Write-Host 'No temp folders to remove.' }
    Write-Host 'Done.'
} catch {
    Write-Host 'Error during prepare_dataset:' $_.Exception.Message
    exit 1
}
