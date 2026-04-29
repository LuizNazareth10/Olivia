Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   OLIVIA PLATFORM DEPLOYMENT SCRIPT V2   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Stop existing processes on ports 8001, 8002, 8080, 7357
Write-Host "Stopping existing services..." -ForegroundColor Yellow
$ports = @(8001, 8002, 8080, 7357)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = $conn | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($p in $pids) {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped process $p on port $port" -ForegroundColor Gray
        }
    }
}

# 2. Start Docker Compose
Write-Host "Starting Docker Containers..." -ForegroundColor Yellow
Set-Location "d:\Olivia\stress-hrv-platform"

# Build services
Write-Host "Building Docker Services..." -ForegroundColor Cyan
docker-compose up -d --build --remove-orphans

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Compose failed. Please ensure Docker Desktop is running." -ForegroundColor Red
    exit
}

Write-Host "Docker Services started." -ForegroundColor Green

# 3. Build and Serve Web App (Mobile Equivalent)
Write-Host "Building Web Version of Mobile App..." -ForegroundColor Yellow
Set-Location "d:\Olivia\stress-hrv-platform\mobile_app"

# Remove old build to force update
if (Test-Path "build\web") {
    Remove-Item "build\web" -Recurse -Force
}

# Run Flutter Web Build (Using generic command compatible with 3.x)
cmd /c "flutter build web --release"

if (Test-Path "build\web\index.html") {
    Write-Host "Starting Web App Server on port 7357..." -ForegroundColor Yellow
    
    # Start Python HTTP Server for Web App in background
    Start-Process -FilePath "python" -ArgumentList "-m http.server 7357 --directory build/web" -WindowStyle Hidden
    
    Start-Sleep -Seconds 2
    
    Write-Host "Web App Server running." -ForegroundColor Green
} else {
    Write-Host "Web build failed. Skipping web app deployment." -ForegroundColor Red
}

# 4. Instructions for Public Access (iPhone)
Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "   OLIVIA PLATFORM - DEPLOYED                    " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "To access the platform on iPhone:"
Write-Host "1. Open a NEW PowerShell terminal"
Write-Host "2. Run this command to create a public link:" -ForegroundColor Yellow
Write-Host "   d:\Olivia\tools\cloudflared.exe tunnel --url http://localhost:7357" -ForegroundColor Cyan
Write-Host "3. Copy the URL ending in .trycloudflare.com and open in Safari"
Write-Host ""
Write-Host "To access the API (if needed):"
Write-Host "   d:\Olivia\tools\cloudflared.exe tunnel --url http://localhost:8001" -ForegroundColor Gray
Write-Host "To access the Dashboard (if needed):"
Write-Host "   d:\Olivia\tools\cloudflared.exe tunnel --url http://localhost:8002" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Green
