# --------------------------------------------------------------
# identify_wing.ps1 - interactive one-shot launcher (Windows)
#
# Windows twin of identify_wing.sh. Two modes:
#   1) Segment a wing photo with SAM3, then identify it
#        - needs: the wing photo, a Hugging Face token for the
#          gated SAM3 model (first download only - the weights are
#          then kept in the 'flywing-sam3-cache' Docker volume),
#          and a prompt-points JSON file (schema below)
#   2) Classify an existing contour CSV (the classic behaviour)
#
# The prompt-points JSON describes where to click: 1 point inside
# each wing cell + 2 points outside it (to exclude neighbours):
#   { "pa2r": {"points": [[x, y], [x, y], [x, y]], "labels": [1, 0, 0]},
#     "dm":   {"points": [[x, y], [x, y], [x, y]], "labels": [1, 0, 0]} }
# The bundled prompts_bkarnyi9.json matches B.karnyi.9_cleanup.png
# and can serve as a template. The GUI wizard (identify_gui.py)
# can generate this file by clicking on the photo.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\identify_wing.ps1
# or simply, if script execution is allowed on your machine:
#   .\identify_wing.ps1
# --------------------------------------------------------------

Set-Location -Path $PSScriptRoot

$Image         = 'flywing-workflow'
$Container     = 'flywing-session'
$SegImage      = 'flywing-segment'
$SegContainer  = 'flywing-segment-session'
$CacheVolume   = 'flywing-sam3-cache'

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
# (the SAM3 segmentation image runs on the host's native architecture)
$arch = docker version --format '{{.Server.Arch}}' 2>$null
$platformFlags = @()
if ($arch -eq 'arm64' -or $arch -eq 'aarch64') {
  $platformFlags = @('--platform', 'linux/amd64')
  Write-Host 'Note: ARM host detected - the x86_64 classification image'
  Write-Host '      runs through emulation; SAM3 runs natively.'
} else {
  Write-Host 'Note: x86_64 host detected - the container runs natively.'
}

# ---- cleanup: remove containers + images when the script ends -----------------
function Cleanup {
  Write-Host ''
  Write-Host '=== Cleaning up ==='
  docker rm -f $Container $SegContainer *> $null
  if (docker image inspect $Image *> $null) {
    docker rmi $Image *> $null
    Write-Host "Classification container and image '$Image' removed."
  }
  if (docker image inspect $SegImage *> $null) {
    docker rmi $SegImage *> $null
    Write-Host "Segmentation container and image '$SegImage' removed."
  }
  Write-Host 'The build cache is kept, so the next start rebuilds quickly.'
  Write-Host "The SAM3 weights volume '$CacheVolume' is kept"
  Write-Host "  (delete with: docker volume rm $CacheVolume)."
  Write-Host 'To also free the build cache: docker builder prune'
}

try {
  # ---- 1. choose the mode -----------------------------------------------------
  Write-Host ''
  Write-Host ' 1) Segment a wing photo with SAM3, then identify it'
  Write-Host ' 2) Classify an existing contour CSV'
  $mode = ''
  while ($true) {
    $mode = Read-Host 'Mode [1/2]'
    if ($mode -eq '1' -or $mode -eq '2') { break }
    Write-Host '  Please type 1 or 2.'
  }

  $base = $null
  if ($mode -eq '1') {
    # ---- 1a. the wing photo ----------------------------------------------------
    $photoBase = $null
    while ($true) {
      $photo = Read-Host 'Wing photo (file name or full path)'
      if ([string]::IsNullOrWhiteSpace($photo)) {
        Write-Host '  Please enter a file name.'; continue
      }
      $photo = $photo.Trim('"').Trim("'")
      if (-not (Test-Path -LiteralPath $photo)) {
        Write-Host "  File not found: $photo"; continue
      }
      $photoBase = Split-Path -Leaf $photo
      $pdir = Split-Path -Parent (Resolve-Path -LiteralPath $photo).Path
      if ($pdir -ne (Get-Location).Path) {
        Copy-Item -LiteralPath $photo -Destination $photoBase
        Write-Host "  Copied into this directory: $photoBase"
      }
      break
    }

    # ---- 1b. Hugging Face token (hidden; ENTER = use cached weights) -----------
    Write-Host ''
    Write-Host 'SAM3 weights are hosted on a gated Hugging Face repo:'
    Write-Host '  request access at https://huggingface.co/facebook/sam3, then paste'
    Write-Host '  your token (https://huggingface.co/settings/tokens).'
    $token = ''
    $secure = Read-Host 'HF token (ENTER = skip, works once the weights are cached)' -AsSecureString
    if ($secure.Length -gt 0) {
      $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
      $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
      [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
      Write-Host '  (the token is passed to the container only - never stored)'
    }

    # ---- 1c. prompt points JSON -------------------------------------------------
    Write-Host ''
    if (Test-Path -LiteralPath 'prompts_bkarnyi9.json') {
      Write-Host "A points file 'prompts_bkarnyi9.json' is bundled for B.karnyi.9_cleanup.png."
    }
    $pointsBase = $null
    while ($true) {
      $points = Read-Host 'Prompt-points JSON (ENTER = prompts_bkarnyi9.json)'
      if ([string]::IsNullOrWhiteSpace($points)) { $points = 'prompts_bkarnyi9.json' }
      $points = $points.Trim('"').Trim("'")
      if (-not (Test-Path -LiteralPath $points)) {
        Write-Host "  File not found: $points (see the schema in the header of identify_wing.ps1)"
        continue
      }
      $pointsBase = Split-Path -Leaf $points
      $pdir = Split-Path -Parent (Resolve-Path -LiteralPath $points).Path
      if ($pdir -ne (Get-Location).Path) {
        Copy-Item -LiteralPath $points -Destination $pointsBase
        Write-Host "  Copied into this directory: $pointsBase"
      }
      break
    }

    # ---- 1d. build + run the segmentation ----------------------------------------
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($photoBase) -replace ' ', '_'
    $outCsv   = "contours_coordinates_$stem.csv"
    $masksDir = "masks_$stem"

    Write-Host ''
    Write-Host '=== Building the SAM3 segmentation image ==='
    docker build -f Dockerfile.segment -t $SegImage .
    if ($LASTEXITCODE -ne 0) { throw 'Segmentation image build failed.' }

    Write-Host ''
    Write-Host '=== Segmenting the wing cells ==='
    Write-Host 'First run downloads ~3.4 GB of gated weights once (kept in the'
    Write-Host "Docker volume '$CacheVolume' for next time)."
    $segArgs = @('--rm', '--name', $SegContainer,
                 '-v', "$((Get-Location).Path):/work", '-w', '/work',
                 '-v', "$CacheVolume`:/root/.cache/huggingface")
    if ($token -ne '') { $segArgs += @('-e', "HF_TOKEN=$token") }
    $segArgs += @($SegImage, $photoBase,
                  '--points', $pointsBase, '--out', $outCsv,
                  '--masks-dir', $masksDir)
    docker run @segArgs
    if ($LASTEXITCODE -ne 0) { throw 'Segmentation run failed.' }

    Write-Host ''
    Write-Host "Quality-check masks saved in '$masksDir' (white = segmented cell)."
    $base = $outCsv
  } else {
    # ---- 2a. prompt for the contour file ---------------------------------------
    Write-Host ''
    Write-Host 'Contour files available in this directory:'
    Get-ChildItem -Path . -Filter 'contours_coordinates*.csv' -Name |
      ForEach-Object { Write-Host "  $_" }
    Write-Host ''

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
  }

  # ---- 3. prompt for the species (ENTER = unknown sample) ---------------------
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

  # ---- 4. build the classification image ---------------------------------------
  Write-Host ''
  Write-Host '=== Building the classification image (cached layers keep this quick) ==='
  docker build @platformFlags -f Dockerfile.workflow -t $Image .
  if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }

  # ---- 5. run the classification -----------------------------------------------
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
