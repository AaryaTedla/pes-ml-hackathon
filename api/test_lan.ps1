param(
    [Parameter(Mandatory = $true)][string]$ApiUrl,
    [string]$TripId
)
$ErrorActionPreference = "Stop"
$baseUrl = $ApiUrl.TrimEnd('/')
$health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 15
$library = Invoke-RestMethod -Uri "$baseUrl/trips" -TimeoutSec 15
if ($health.status -ne "ok" -or $library.trips.Count -eq 0) { throw "API is not ready or has no trips." }
if (-not $TripId) { $TripId = $library.trips[0] }
$payload = @{ trip_id = $TripId } | ConvertTo-Json
$result = Invoke-RestMethod -Uri "$baseUrl/predict" -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 120
if ($result.trip_id -ne $TripId -or $result.gps.Count -ne $result.n_points -or $result.predicted_arcs.Count -ne $result.n_points) { throw "Invalid prediction response." }
if ($result.PSObject.Properties.Name -contains "reference_route" -or $result.PSObject.Properties.Name -contains "true_arcs") { throw "Reference information must not reach laptop 2." }
Write-Host "PASS | API=$baseUrl | trips=$($library.trips.Count) | trip=$TripId | points=$($result.n_points) | time=$($result.timing_ms.total)ms | accuracy=$($result.metrics.pointwise_accuracy) | mode=$($result.mode)"
Write-Host "Browser check: open $baseUrl/ui/, run the trip, toggle both map layers, and check the API terminal log."
