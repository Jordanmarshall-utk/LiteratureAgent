param(
    [string]$Destination = "",
    [switch]$IncludeBaselineDatabase,
    [switch]$IncludeCurrentStructuredOutputs,
    [switch]$IncludeCredentials,
    [string[]]$ExistingOutputDirs = @()
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $Destination) {
    $Destination = Join-Path $project "deliverables\standalone_packages\LiteratureAgent_standalone_platform_$stamp.zip"
}

$stage = Join-Path $env:TEMP "LiteratureAgent_standalone_platform_$stamp"
$packageRoot = Join-Path $stage "LiteratureAgent"
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

$files = @(
    "literature_agent_full_end_to_end_v21_3_english_sanitizer.py",
    "README.md",
    "requirements.txt",
    "requirements-platform.txt",
    "Setup_LiteratureAgent.cmd",
    "Launch_LiteratureAgent.cmd",
    "START_LITERATURE_AGENT.cmd",
    ".gitignore"
)
$folders = @("literature_agent", "literature_agent_platform", "models", "config", "docs", "scripts")

foreach ($file in $files) {
    Copy-Item -LiteralPath (Join-Path $project $file) -Destination $packageRoot -Force
}
foreach ($folder in $folders) {
    Copy-Item -LiteralPath (Join-Path $project $folder) -Destination $packageRoot -Recurse -Force
}
Get-ChildItem -LiteralPath $packageRoot -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $packageRoot -File -Recurse |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force

New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "data") | Out-Null
if ($IncludeBaselineDatabase) {
    Copy-Item -LiteralPath (Join-Path $project "data\Perovskite_database_content_all_data.csv") -Destination (Join-Path $packageRoot "data") -Force
} else {
    Set-Content -LiteralPath (Join-Path $packageRoot "data\PLACE_BASELINE_DATABASE_HERE.txt") -Value "Place Perovskite_database_content_all_data.csv in this folder before running integration or modeling."
}
if (Test-Path -LiteralPath (Join-Path $project "data\expansion_processed_papers")) {
    Copy-Item -LiteralPath (Join-Path $project "data\expansion_processed_papers") -Destination (Join-Path $packageRoot "data") -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Join-Path $packageRoot "secrets") | Out-Null
Copy-Item -LiteralPath (Join-Path $project "secrets\README.md") -Destination (Join-Path $packageRoot "secrets") -Force
if ($IncludeCredentials) {
    $credentialFiles = @("google_drive_oauth_client.json", "google_drive_token.json")
    foreach ($credentialFile in $credentialFiles) {
        $credentialPath = Join-Path $project "secrets\$credentialFile"
        if (-not (Test-Path -LiteralPath $credentialPath)) {
            throw "Required credential file was not found: $credentialPath"
        }
        Copy-Item -LiteralPath $credentialPath -Destination (Join-Path $packageRoot "secrets") -Force
    }
}

if ($IncludeCurrentStructuredOutputs) {
    if ($ExistingOutputDirs.Count -eq 0) {
        throw "-IncludeCurrentStructuredOutputs requires one or more explicit -ExistingOutputDirs paths."
    }
    $bundledRoot = Join-Path $packageRoot "bundled_collections"
    New-Item -ItemType Directory -Force -Path $bundledRoot | Out-Null
    $structuredFolders = @(
        "csv", "paper_summaries_json", "paper_summaries_text", "figure_reports",
        "vision_logs", "qa_reports", "timing_logs"
    )
    foreach ($source in $ExistingOutputDirs) {
        if (-not (Test-Path -LiteralPath (Join-Path $source "csv\all_records.csv"))) { continue }
        $target = Join-Path $bundledRoot (Split-Path -Leaf $source)
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        foreach ($folder in $structuredFolders) {
            $sourceFolder = Join-Path $source $folder
            if (Test-Path -LiteralPath $sourceFolder) {
                Copy-Item -LiteralPath $sourceFolder -Destination $target -Recurse -Force
            }
        }
        Get-ChildItem -LiteralPath $source -File |
            Where-Object { $_.Extension -in @(".json", ".csv", ".md", ".txt") } |
            Copy-Item -Destination $target -Force
    }
    $existingKnowledgeGraphs = Join-Path $project "artifacts\literature_agent_pce_stability\source_data\knowledge_graph\all_policy_comparison"
    if (Test-Path -LiteralPath $existingKnowledgeGraphs) {
        $bundledResults = Join-Path $packageRoot "bundled_results"
        New-Item -ItemType Directory -Force -Path $bundledResults | Out-Null
        $policyBundle = Join-Path $bundledResults "all_policy_comparison"
        New-Item -ItemType Directory -Force -Path $policyBundle | Out-Null
        foreach ($rootFile in @("README.txt", "reasoning_policy_kg_comparison.csv")) {
            $sourceFile = Join-Path $existingKnowledgeGraphs $rootFile
            if (Test-Path -LiteralPath $sourceFile) {
                Copy-Item -LiteralPath $sourceFile -Destination $policyBundle -Force
            }
        }
        $knowledgeGraphFolders = @("neo4j_import", "reports", "derived_views")
        foreach ($variant in Get-ChildItem -LiteralPath $existingKnowledgeGraphs -Directory) {
            $variantTarget = Join-Path $policyBundle $variant.Name
            foreach ($folder in $knowledgeGraphFolders) {
                $sourceFolder = Join-Path $variant.FullName $folder
                if (Test-Path -LiteralPath $sourceFolder) {
                    New-Item -ItemType Directory -Force -Path $variantTarget | Out-Null
                    Copy-Item -LiteralPath $sourceFolder -Destination $variantTarget -Recurse -Force
                }
            }
            $variantReadme = Join-Path $variant.FullName "README.txt"
            if (Test-Path -LiteralPath $variantReadme) {
                New-Item -ItemType Directory -Force -Path $variantTarget | Out-Null
                Copy-Item -LiteralPath $variantReadme -Destination $variantTarget -Force
            }
        }
    }
}

$packageReadme = @"
# LiteratureAgent Standalone Platform

1. Extract this folder to a local drive.
2. Install Ollama from https://ollama.com/download if it is not already installed.
3. Double-click `START_LITERATURE_AGENT.cmd`.
4. The first launch installs Python dependencies and downloads required local models.
5. Open Home and confirm the system checks.

Downloaded papers and caches are intentionally excluded. Structured prior outputs are included only with `-IncludeCurrentStructuredOutputs`. Bundled reasoning-policy graphs include Neo4j imports, reports, and derived views; duplicate portable JSONL and GraphML exports are excluded. Google credentials are included only with `-IncludeCredentials`; treat that ZIP as private. See `docs/STANDALONE_PLATFORM_GUIDE.md`.
"@
Set-Content -LiteralPath (Join-Path $packageRoot "START_HERE.md") -Value $packageReadme -Encoding UTF8

$destinationParent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Force }
Compress-Archive -LiteralPath $packageRoot -DestinationPath $Destination -CompressionLevel Optimal

$entries = [System.IO.Compression.ZipFile]::OpenRead($Destination)
$count = $entries.Entries.Count
$entries.Dispose()
Remove-Item -LiteralPath $stage -Recurse -Force

Write-Host "Created: $Destination"
Write-Host "Files in package: $count"
