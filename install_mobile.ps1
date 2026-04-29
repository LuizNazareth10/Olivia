Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   OLIVIA MOBILE APP INSTALLER            " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Flutter
if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Write-Host "Flutter is required but not found in PATH." -ForegroundColor Red
    exit 1
}

# Check Devices
Write-Host "Checking for connected Android devices..." -ForegroundColor Yellow
$devices = flutter devices
if ($devices -notmatch "android") {
    Write-Host "No Android device detected. Please connect via USB and enable USB Debugging." -ForegroundColor Red
    Write-Host "Or start an emulator." -ForegroundColor Gray
    exit 1
}

Write-Host "Device found. Building Release APK..." -ForegroundColor Yellow
Set-Location "d:\Olivia\stress-hrv-platform\mobile_app"

# Build
flutter build apk --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

$apkPath = "build\app\outputs\flutter-apk\app-release.apk"
if (Test-Path $apkPath) {
    Write-Host "Installing APK to device..." -ForegroundColor Yellow
    flutter install
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS! App installed." -ForegroundColor Green
        Write-Host "Run the app and enter the Edge Service URL from the start_platform script." -ForegroundColor Cyan
    } else {
        Write-Host "Installation failed. Try running 'flutter run --release' manually." -ForegroundColor Red
    }
} else {
    Write-Host "APK not found at $apkPath" -ForegroundColor Red
}
