param(
    [switch]$SetupOnly,
    [ValidateSet("auto", "laptop", "quality")]
    [string]$ModelProfile = "auto",
    [switch]$SkipModelSmokeTest
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

if (-not $env:LITERATUREAGENT_RUNTIME_HOME) {
    $env:LITERATUREAGENT_RUNTIME_HOME = Join-Path $env:LOCALAPPDATA "LiteratureAgent"
}
$runtimeHome = $env:LITERATUREAGENT_RUNTIME_HOME
$venv = Join-Path $runtimeHome "venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$statePath = Join-Path $runtimeHome "install_state.json"
if ($ModelProfile -eq "auto") {
    if (Test-Path -LiteralPath $statePath) {
        try { $ModelProfile = (Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json).model_profile } catch {}
    }
    if ($ModelProfile -notin @("laptop", "quality")) { $ModelProfile = "laptop" }
}

function Find-Python {
    $bundled = Join-Path $project ".tools\Python312\python.exe"
    if (Test-Path -LiteralPath $bundled) { return @{Exe=$bundled; Prefix=@()} }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return @{Exe=$py.Source; Prefix=@("-3.11")} }
    foreach ($candidate in @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) { return @{Exe=$candidate; Prefix=@()} }
    }
    return $null
}

function Find-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-OllamaApi([int]$Seconds = 90) {
    for ($attempt = 0; $attempt -lt $Seconds; $attempt++) {
        if (Test-OllamaApi) { return }
        Start-Sleep -Seconds 1
    }
    throw "Ollama did not become ready at http://127.0.0.1:11434 within $Seconds seconds."
}

function Ensure-OllamaModel([string]$OllamaExe, [string]$ModelName) {
    $modelExists = $false
    try {
        & $OllamaExe show $ModelName *> $null
        $modelExists = $LASTEXITCODE -eq 0
    } catch {
        $modelExists = $false
    }
    if ($modelExists) {
        Write-Host "Model ready: $ModelName" -ForegroundColor Green
        return
    }
    Write-Host "Downloading $ModelName. This is a large first-run download..." -ForegroundColor Cyan
    & $OllamaExe pull $ModelName
    if ($LASTEXITCODE -ne 0) { throw "Ollama could not download $ModelName." }
    & $OllamaExe show $ModelName *> $null
    if ($LASTEXITCODE -ne 0) { throw "Ollama downloaded $ModelName but could not verify it." }
}

Write-Host "" 
Write-Host "LiteratureAgent supported Windows setup" -ForegroundColor Cyan
Write-Host "Project: $project"
Write-Host "Private runtime: $runtimeHome"
Write-Host "Model profile: $ModelProfile"
Write-Host ""

New-Item -ItemType Directory -Force -Path $runtimeHome | Out-Null
$python = Find-Python
if (-not $python) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python is missing. Re-extract the complete bundle, which includes .tools\Python312."
    }
    Write-Host "Installing Python 3.11 with Windows Package Manager..."
    & $winget.Source install --exact --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python installation failed." }
    $python = Find-Python
    if (-not $python) { throw "Python was installed but not found. Restart Windows, then run setup again." }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "[1/5] Creating the private Python environment..."
    & $python.Exe @($python.Prefix) -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw "Could not create the private Python environment at $venv."
    }
} else {
    Write-Host "[1/5] Private Python environment already exists."
}

$requirements = Join-Path $project "requirements-platform.txt"
$requirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $requirements).Hash
$installedHash = ""
if (Test-Path -LiteralPath $statePath) {
    try { $installedHash = (Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json).requirements_sha256 } catch {}
}
$pythonReady = $false
try {
    & $venvPython -c "import streamlit" *> $null
    $pythonReady = $LASTEXITCODE -eq 0
} catch {
    $pythonReady = $false
}
if (-not $pythonReady -or $installedHash -ne $requirementsHash) {
    Write-Host "[2/5] Installing LiteratureAgent Python dependencies..."
    & $venvPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. Check the internet connection and retry." }
    & $venvPython -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed. Check the internet connection and retry." }
} else {
    Write-Host "[2/5] Python dependencies are ready."
}

$ollama = Find-Ollama
if (-not $ollama) {
    $bundledInstaller = Join-Path $project "installers\OllamaSetup.exe"
    if (Test-Path -LiteralPath $bundledInstaller) {
        Write-Host "[3/5] Opening the bundled official Ollama installer..." -ForegroundColor Cyan
        Write-Host "Complete the Ollama installer window; setup will continue automatically."
        $installerProcess = Start-Process -FilePath $bundledInstaller -Wait -PassThru
        if ($installerProcess.ExitCode -ne 0) { throw "The Ollama installer exited with code $($installerProcess.ExitCode)." }
    } else {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) { throw "Ollama is missing and installers\OllamaSetup.exe is not present." }
        Write-Host "[3/5] Installing Ollama with Windows Package Manager..."
        & $winget.Source install --exact --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Ollama installation failed." }
    }
    $ollama = Find-Ollama
    if (-not $ollama) { throw "Ollama was installed but ollama.exe was not found. Restart Windows, then run setup again." }
} else {
    Write-Host "[3/5] Ollama is installed: $ollama"
}

if (-not (Test-OllamaApi)) {
    Write-Host "Starting the local Ollama service..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden | Out-Null
}
Wait-OllamaApi

if ($ModelProfile -eq "quality") {
    $textModel = "qwen2.5:7b"
    $visionModel = "qwen2.5vl:7b-q4_K_M"
} else {
    $textModel = "qwen2.5:3b"
    $visionModel = "qwen2.5vl:3b"
}
Write-Host "[4/5] Verifying local Qwen models..."
Ensure-OllamaModel $ollama $textModel
Ensure-OllamaModel $ollama $visionModel

if (-not $SkipModelSmokeTest) {
    Write-Host "Testing the local text model..."
    $body = @{model=$textModel; stream=$false; messages=@(@{role="user"; content="Reply with OK."}); options=@{num_predict=8}} | ConvertTo-Json -Depth 6
    $smoke = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/chat" -ContentType "application/json" -Body $body -TimeoutSec 180
    if (-not $smoke.message.content) { throw "The installed text model did not return a smoke-test response." }
}

$settings = [ordered]@{
    base_csv = (Join-Path $project "data\Perovskite_database_content_all_data.csv")
    ontology = (Join-Path $project "config\domain_packs\general_science.json")
    llm_api_url = "http://localhost:11434/v1/chat/completions"
    llm_model = $textModel
    vision_api_url = "http://localhost:11434/v1/chat/completions"
    vision_model = $visionModel
    oauth_client = (Join-Path $project "secrets\google_drive_oauth_client.json")
    oauth_token = (Join-Path $project "secrets\google_drive_token.json")
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $runtimeHome "settings.json"),
    ($settings | ConvertTo-Json -Depth 4),
    $utf8NoBom
)
$state = [ordered]@{
    package_version = "2.0.0"
    requirements_sha256 = $requirementsHash
    model_profile = $ModelProfile
    text_model = $textModel
    vision_model = $visionModel
    installed_at = (Get-Date).ToString("o")
}
[System.IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json), $utf8NoBom)
Write-Host "[5/5] Setup verification passed." -ForegroundColor Green

if ($SetupOnly -or $env:LITERATUREAGENT_SETUP_ONLY -eq "1") {
    Write-Host "LiteratureAgent setup is complete." -ForegroundColor Green
    exit 0
}

$platformUrl = "http://127.0.0.1:8501"
$platformHealthy = $false
try {
    Invoke-WebRequest -UseBasicParsing "$platformUrl/_stcore/health" -TimeoutSec 2 | Out-Null
    $platformHealthy = $true
} catch {}

if (-not $platformHealthy) {
    Write-Host "Starting LiteratureAgent..."
    $stdoutLog = Join-Path $runtimeHome "server.stdout.log"
    $stderrLog = Join-Path $runtimeHome "server.stderr.log"
    $streamlitArgs = @(
        "-m", "streamlit", "run", "literature_agent_platform\app.py",
        "--server.address", "127.0.0.1", "--server.port", "8501",
        "--server.headless", "true", "--browser.gatherUsageStats", "false"
    )
    Start-Process -FilePath $venvPython -ArgumentList $streamlitArgs -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog | Out-Null
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-WebRequest -UseBasicParsing "$platformUrl/_stcore/health" -TimeoutSec 2 | Out-Null
            $platformHealthy = $true
            break
        } catch {}
    }
}

if (-not $platformHealthy) {
    throw "LiteratureAgent did not become ready. Review $runtimeHome\server.stderr.log."
}
Write-Host "LiteratureAgent is ready: $platformUrl" -ForegroundColor Green
if ($env:LITERATUREAGENT_NO_BROWSER -ne "1") { Start-Process $platformUrl }
