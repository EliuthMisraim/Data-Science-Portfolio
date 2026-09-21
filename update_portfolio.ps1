# Script to update the Data Science Portfolio repository
# usage: ./update_portfolio.ps1 ["Optional Commit Message"]

param (
    [string]$Message = "Update portfolio: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

$repoPath = if ($PSScriptRoot) { $PSScriptRoot } else { "c:\Users\eliut\Downloads\Data Science stuff\Data-Science-Portfolio" }

# Verify path exists
if (-not (Test-Path $repoPath)) {
    Write-Host "Error: Repository path not found at $repoPath" -ForegroundColor Red
    Pause
    exit 1
}

# Navigate to repo
Set-Location -Path $repoPath

# Check if inside git work tree
$isGitRepo = git rev-parse --is-inside-work-tree 2>$null
if ($isGitRepo -ne "true") {
    Write-Host "Error: $repoPath is not a valid git repository." -ForegroundColor Red
    Pause
    exit 1
}

# Check for pending merge
$gitDir = git rev-parse --git-dir
$mergeHeadPath = Join-Path $gitDir "MERGE_HEAD"
if (Test-Path $mergeHeadPath) {
    Write-Host "Detected an ongoing merge..." -ForegroundColor Yellow
    $unmergedFiles = git diff --name-only --diff-filter=U
    if ($unmergedFiles) {
        Write-Host "Conflict detected in files:" -ForegroundColor Red
        Write-Host $unmergedFiles -ForegroundColor Red
        Write-Host "Please resolve conflicts before continuing." -ForegroundColor Red
        Pause
        exit 1
    } else {
        Write-Host "Completing pending merge commit..." -ForegroundColor Cyan
        git commit --no-edit -m "Merge remote-tracking branch 'origin/main'"
    }
} else {
    # Pull latest changes to avoid conflicts
    Write-Host "Pulling latest changes from GitHub..." -ForegroundColor Cyan
    git pull origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: git pull encountered an issue. Check messages above." -ForegroundColor Yellow
    }
}

# Add all changes
Write-Host "Checking local changes..." -ForegroundColor Cyan
git add .

# Check for changes to commit
$status = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($status)) {
    Write-Host "Files to be updated:" -ForegroundColor Green
    Write-Host $status
    Write-Host "Committing with message: '$Message'" -ForegroundColor Cyan
    git commit -m $Message
} else {
    Write-Host "No new file changes to commit." -ForegroundColor Yellow
}

# Check if we have unpushed commits (ahead of origin/main)
$unpushed = git log origin/main..HEAD --oneline 2>$null
if (-not [string]::IsNullOrWhiteSpace($unpushed)) {
    Write-Host "Commits pending to push:" -ForegroundColor Cyan
    Write-Host $unpushed
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully updated and pushed portfolio to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "Error pushing changes to GitHub." -ForegroundColor Red
    }
} else {
    Write-Host "Portfolio is fully up to date with GitHub." -ForegroundColor Green
}

Pause

