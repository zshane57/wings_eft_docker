# --------------------------------------------------------------
# start_web.ps1 - start the browser frontend (flywing-web container)
#
# Windows twin of start_web.sh: builds the small web-server image
# (cached after the first time), starts it with the Docker socket
# and this package directory mounted, and opens the page:
#
#     http://localhost:8080
#
# The server keeps running until you stop it:
#     docker rm -f flywing-web
# (the SAM3 weights volume and the build caches are kept)
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\start_web.ps1
# --------------------------------------------------------------

Set-Location -Path $PSScriptRoot

$Image = 'flywing-web'
$Port  = 8080

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host 'Error: Docker was not found on this machine.'
  exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host 'Error: Docker is not running. Start Docker Desktop first, then re-run.'
  exit 1
}

Write-Host '=== Building the web frontend image (cached layers keep this quick) ==='
docker build -f Dockerfile.web -t $Image .
if ($LASTEXITCODE -ne 0) { exit 1 }

docker rm -f $Image *> $null
$workPath = (Get-Location).Path
docker run -d --name $Image `
  -v /var/run/docker.sock:/var/run/docker.sock `
  -v "$workPath`:/app/work" `
  -e "WORK_HOST_PATH=$workPath" `
  -p "127.0.0.1:$Port`:$Port" `
  $Image
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ''
Write-Host "Web frontend running at http://localhost:$Port"
Write-Host "Stop it with:   docker rm -f $Image"
Start-Process "http://localhost:$Port"
