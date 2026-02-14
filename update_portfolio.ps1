# Script to update the Data Science Portfolio repository
# usage: ./update_portfolio.ps1 ["Optional Commit Message"]

param (
    [string]$Message = "Update portfolio: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

$repoPath = "c:\Users\eliut\Downloads\Data Science stuff\Data-Science-Portfolio"

# Verify path exists
if (-not (Test-Path $repoPath)) {
    Write-Host "Error: Repository path not found at $repoPath" -ForegroundColor Red
    exit 1
}

# Navigate to repo
Set-Location -Path $repoPath

# Pull latest changes to avoid conflicts
Write-Host "Pulling latest changes..." -ForegroundColor Cyan
git pull origin main

# Add all changes
Write-Host "Adding changes..." -ForegroundColor Cyan
git add .

# Check for changes to commit
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No changes to commit." -ForegroundColor Yellow
    exit 0
}

# Commit
Write-Host "Committing with message: '$Message'" -ForegroundColor Cyan
git commit -m $Message

# Push
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully updated portfolio!" -ForegroundColor Green
} else {
    Write-Host "Error pushing changes." -ForegroundColor Red
}

Pause
