param (
    [string]$Username = "Manoj",
    [string]$Dataset = "Synthetic_patient-HealthCare-Monitoring_dataset.csv",
    [string]$Target = "Predicted Disease",
    [string]$Task = "classification",
    [int]$Clients = 5,
    [int]$Rounds = 10,
    [string]$Distribution = "iid"
)

$DataPath = Join-Path "data" $Dataset
if (-not (Test-Path $DataPath)) {
    Write-Error "Dataset not found at $DataPath. Ensure it exists in the data/ folder."
    exit 1
}

$PythonCmd = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PythonCmd = ".venv\Scripts\python.exe"
} elseif (Test-Path ".fed\Scripts\python.exe") {
    $PythonCmd = ".fed\Scripts\python.exe"
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Starting FL Simulation for user: $Username" -ForegroundColor Cyan
Write-Host "Dataset: $DataPath | Target: $Target | Clients: $Clients | Rounds: $Rounds" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

& $PythonCmd fl_simulation.py `
    --data_path "$DataPath" `
    --target_col "$Target" `
    --username "$Username" `
    --task_type "$Task" `
    --distribution "$Distribution" `
    --num_clients $Clients `
    --num_rounds $Rounds

if ($LASTEXITCODE -eq 0) {
    Write-Host "Simulation complete! Generating reports..." -ForegroundColor Green
    $LatestJson = Get-ChildItem -Path "results" -Filter "simulation_results_${Username}_*.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($LatestJson) {
        & $PythonCmd generate_report.py --result_file $LatestJson.FullName
        Write-Host "==================================================" -ForegroundColor Green
        Write-Host "All Done! Check the reports/ folder." -ForegroundColor Green
    } else {
        Write-Warning "Could not locate generated results JSON in results/ folder."
    }
} else {
    Write-Error "Simulation failed with exit code $LASTEXITCODE."
}
