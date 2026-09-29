# --------------------------------------------------------------
# identify_wing.ps1 - interactive one-shot launcher (Windows)
#
# Windows twin of identify_wing.sh: prompts for the contour CSV and
# (optionally) the sample's species, builds the Docker image, runs the
# classification workflow, and afterwards removes the container and
# the image again.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\identify_wing.ps1
# or simply, if script execution is allowed on your machine:
#   .\identify_wing.ps1
#
# Nothing else is required: files named contours_coordinates*.csv in
# this directory are baked into the image at build time.
# --------------------------------------------------------------

Set-Location -Path $PSScriptRoot

$Image     = 'flywing-workflow'
$Container = 'flywing-session'

$ValidSpecies = @(
  'C.vicina', 'Ch.albiceps', 'Ch.bezziana', 'Ch.megacephala', 'Ch.nigripes',
  'Ch.rufifacies', 'L.sericata', 'A.gressitti', 'B.karnyi', 'Le.alba',
  'S.princeps', 'Sy.nudiseta', 'Z.aquila'
)

Write-Host '================================================================'
Write-Host ' Forensic fly wing species identification'
Write-Host '================================================================'

# ---- 0. sanity checks --------------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host 'Error: Docker was not found on this machine.'
  exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host 'Error: Docker is not running. Start Docker Desktop first, then re-run.'
  exit 1
}

# The GUIDE binary is x86_64 only - emulate on ARM hosts
$arch = docker version --format '{{.Server.Arch}}' 2>$null
$platformFlags = @()
if ($arch -eq 'arm64' -or $arch -eq 'aarch64') {
  $platformFlags = @('--platform', 'linux/amd64')
  Write-Host 'Note: ARM host detected - the x86_64 image runs through emulation.'
} else {
  Write-Host 'Note: x86_64 host detected - the container runs natively.'
}

# ---- cleanup: remove container + image when the script ends -------------------
function Cleanup {
  Write-Host ''
  Write-Host '=== Cleaning up ==='
  docker rm -f $Container *> $null
  if (docker image inspect $Image *> $null) {
    docker rmi $Image *> $null
    Write-Host "Container and image '$Image' removed."
  } else {
    Write-Host 'Container removed.'
  }
  Write-Host 'The build cache is kept, so the next start rebuilds quickly.'
  Write-Host 'To also free the build cache: docker builder prune'
}

try {
  # ---- 1. prompt for the contour file ----------------------------------------
  Write-Host ''
  Write-Host 'Contour files available in this directory:'
  Get-ChildItem -Path . -Filter 'contours_coordinates*.csv' -Name |
    ForEach-Object { Write-Host "  $_" }
  Write-Host ''

  $base = $null
  while ($true) {
    $contours = Read-Host 'Contour CSV to identify (file name or full path)'
    if ([string]::IsNullOrWhiteSpace($contours)) {
      Write-Host '  Please enter a file name.'; continue
    }
    # paths pasted from Windows Explorer often arrive wrapped in quotes
    $contours = $contours.Trim('"').Trim("'")
    if (-not (Test-Path -LiteralPath $contours)) {
      Write-Host "  File not found: $contours"; continue
    }

    $base = Split-Path -Leaf $contours
    $dir  = Split-Path -Parent (Resolve-Path -LiteralPath $contours).Path
    if ($base -like 'contours_coordinates*.csv') {
      if ($dir -ne (Get-Location).Path) {
        Copy-Item -LiteralPath $contours -Destination $base
        Write-Host "  Copied into this directory: $base"
      }
      break
    } else {
      # the image build only picks up files named contours_coordinates*.csv
      $dest = 'contours_coordinates_' +
              [System.IO.Path]::GetFileNameWithoutExtension($base) + '.csv'
      Copy-Item -LiteralPath $contours -Destination $dest
      $base = $dest
      Write-Host "  Renamed to $base (image build needs the contours_coordinates*.csv naming)."
      break
    }
  }

  # ---- 2. prompt for the species (ENTER = unknown sample) ---------------------
  Write-Host ''
  Write-Host ('Training species: ' + ($ValidSpecies -join ' '))
  Write-Host '(Press ENTER without typing anything if the sample is unknown -'
  Write-Host ' the workflow then reports the top-3 most probable species.)'

  $species = ''
  while ($true) {
    $answer = Read-Host 'Species of the sample (ENTER = unknown)'
    if ([string]::IsNullOrWhiteSpace($answer)) { break }
    if ($ValidSpecies -contains $answer) { $species = $answer; break }
    Write-Host "  '$answer' is not one of the training species - check the spelling (list above)."
  }

  # ---- 3. build the image --------------------------------------------------------
  Write-Host ''
  Write-Host '=== Building the Docker image (cached layers keep this quick) ==='
  docker build @platformFlags -f Dockerfile.workflow -t $Image .
  if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }

  # ---- 4. run the classification -------------------------------------------------
  Write-Host ''
  if ($species -ne '') {
    docker run --rm --name $Container @platformFlags $Image $base $species
  } else {
    docker run --rm --name $Container @platformFlags $Image $base
  }
  if ($LASTEXITCODE -ne 0) { throw 'Classification run failed.' }
} finally {
  Cleanup
}
