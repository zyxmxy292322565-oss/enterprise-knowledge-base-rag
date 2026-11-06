param(
    [Parameter(Mandatory)] [ValidateSet('Start', 'Stop', 'Status')] [string] $Action,
    [Parameter(Mandatory)] [string] $EnvFile,
    [string] $Docker = 'docker'
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$configuration = (Resolve-Path -LiteralPath $EnvFile).Path
$compose = @('compose', '--project-name', 'ragv3app20261106', '--project-directory', $repository,
    '--env-file', $configuration, '-f', (Join-Path $repository 'compose.full-isolated.yml'))
$services = @('mysql', 'redis', 'api', 'ui')
switch ($Action) {
    'Start' { & $Docker @compose up -d --wait --wait-timeout 180 @services }
    'Stop' { & $Docker @compose stop ui api redis mysql }
    'Status' { & $Docker @compose ps -a @services }
}
if ($LASTEXITCODE -ne 0) {
    throw "RAG V3 isolated stack operation failed with exit code $LASTEXITCODE"
}
