$ErrorActionPreference = 'Stop'
$frontend = Split-Path -Parent $PSScriptRoot
$previousApiBaseUrl = $env:VITE_API_BASE_URL
$api = Start-Process -FilePath 'node.exe' -ArgumentList 'e2e/mock-api.mjs' -WorkingDirectory $frontend -WindowStyle Hidden -PassThru
$env:VITE_API_BASE_URL = 'http://127.0.0.1:18000'
$vite = Start-Process -FilePath 'node.exe' -ArgumentList './node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4174' -WorkingDirectory $frontend -WindowStyle Hidden -PassThru

try {
    foreach ($url in 'http://127.0.0.1:18000/health', 'http://127.0.0.1:4174') {
        $ready = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            try {
                Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 1 | Out-Null
                $ready = $true
                break
            } catch {
                Start-Sleep -Milliseconds 250
            }
        }
        if (-not $ready) { throw "E2E service did not become ready: $url" }
    }
    & npx.cmd playwright test
    $exitCode = $LASTEXITCODE
} finally {
    Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $vite.Id -ErrorAction SilentlyContinue
    $env:VITE_API_BASE_URL = $previousApiBaseUrl
}

exit $exitCode
