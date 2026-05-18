param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectId,

  [Parameter(Mandatory = $true)]
  [string]$BucketName,

  [string]$Region = "us-central1"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Setting active project to $ProjectId"
gcloud config set project $ProjectId | Out-Null

Write-Host "Ensuring bucket gs://$BucketName exists"
$bucketExists = $true
try {
  gsutil ls -b "gs://$BucketName" | Out-Null
} catch {
  $bucketExists = $false
}

if (-not $bucketExists) {
  gcloud storage buckets create "gs://$BucketName" --project=$ProjectId --location=$Region --uniform-bucket-level-access
}

Write-Host "Enabling static website config"
gsutil web set -m index.html -e index.html "gs://$BucketName"

Write-Host "Making site publicly readable"
gsutil iam ch allUsers:objectViewer "gs://$BucketName"

Write-Host "Uploading frontend files"
gsutil -m rsync -r $scriptDir "gs://$BucketName"

Write-Host "Deployment complete"
Write-Host "Website URL: http://$BucketName.storage.googleapis.com/index.html"
