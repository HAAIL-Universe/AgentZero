# pushme.ps1
param(
    [string[]]$files = @("core/llm_wrapper.py", "simulate_test_run.py"),
    [string]$message = "Quick update"
)

# Reset staging
git reset

# Add only the specific files
foreach ($file in $files) {
    git add $file
}

# Show status for review
git status

# Commit and push
git commit -m $message
git push origin main
