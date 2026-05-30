param(
    [string]$SourceDir = '',
    [string]$OutputDir = '',
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Path $PSScriptRoot -Parent

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $repoRoot 'output\wikipedia-pets\texts'
    $nestedSourceDir = Join-Path $repoRoot 'output\wikipedia-pets\wikipedia-pets\texts'
    if ((-not (Test-Path -LiteralPath $SourceDir)) -and (Test-Path -LiteralPath $nestedSourceDir)) {
        $SourceDir = $nestedSourceDir
    }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot 'output\wikipedia-pets\texts-cleaned'
}

$trailingSectionHeadings = @(
    'References',
    'Notes',
    'Footnotes',
    'Citations',
    'Sources',
    'Bibliography',
    'Works cited',
    'Further reading',
    'External links'
)

function Split-ExportedTextDocument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $match = [regex]::Match(
        $Content,
        '^(?<header>.*?(?:\r?\n){2})(?<body>.*)$',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )

    if (-not $match.Success) {
        return [pscustomobject]@{
            Header = ''
            Body   = $Content
        }
    }

    return [pscustomobject]@{
        Header = $match.Groups['header'].Value
        Body   = $match.Groups['body'].Value
    }
}

function Get-DocumentTitleFromHeader {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Header
    )

    $match = [regex]::Match($Header, '^Title:\s*(.+)$', [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($match.Success) {
        return $match.Groups[1].Value.Trim()
    }

    return ''
}

function Remove-TrailingClutterSections {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [AllowEmptyString()]
        [string[]]$Lines
    )

    for ($index = 0; $index -lt $Lines.Count; $index++) {
        $trimmed = $Lines[$index].Trim()
        if ($trailingSectionHeadings -contains $trimmed) {
            if ($index -eq 0) {
                return @()
            }

            return @($Lines[0..($index - 1)])
        }
    }

    return @($Lines)
}

function Normalize-BodyText {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Body
    )

    if ([string]::IsNullOrWhiteSpace($Body)) {
        return ''
    }

    $lines = $Body -split '\r?\n'
    $withoutEditMarkers = New-Object System.Collections.Generic.List[string]

    foreach ($line in $lines) {
        if ($line.Trim() -ceq 'edit') {
            continue
        }

        [void]$withoutEditMarkers.Add($line.TrimEnd())
    }

    $contentLines = Remove-TrailingClutterSections -Lines $withoutEditMarkers.ToArray()
    $normalizedBody = ($contentLines -join "`r`n").Trim()
    $normalizedBody = [regex]::Replace($normalizedBody, '(\r?\n){3,}', "`r`n`r`n")

    return $normalizedBody
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Source directory '$SourceDir' does not exist."
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$files = Get-ChildItem -LiteralPath $SourceDir -File -Filter '*.txt' | Sort-Object Name

if ($files.Count -eq 0) {
    throw "No .txt files were found in '$SourceDir'."
}

$writtenCount = 0
$skippedCount = 0

foreach ($file in $files) {
    $targetPath = Join-Path $OutputDir $file.Name
    if ($SkipExisting -and (Test-Path -LiteralPath $targetPath)) {
        $skippedCount++
        continue
    }

    $rawContent = Get-Content -LiteralPath $file.FullName -Raw
    $parts = Split-ExportedTextDocument -Content $rawContent
    $title = Get-DocumentTitleFromHeader -Header $parts.Header
    $cleanBody = Normalize-BodyText -Body $parts.Body
    $cleanSections = New-Object System.Collections.Generic.List[string]

    if (-not [string]::IsNullOrWhiteSpace($title)) {
        [void]$cleanSections.Add($title)
    }

    if (-not [string]::IsNullOrWhiteSpace($cleanBody)) {
        [void]$cleanSections.Add($cleanBody)
    }

    $cleanContent = (($cleanSections.ToArray()) -join "`r`n`r`n").TrimEnd() + "`r`n"

    Set-Content -LiteralPath $targetPath -Value $cleanContent -Encoding UTF8
    $writtenCount++
}

Write-Host ("Cleaned {0} file(s)." -f $writtenCount)
if ($SkipExisting) {
    Write-Host ("Skipped {0} existing file(s)." -f $skippedCount)
}
Write-Host "  Source: $SourceDir"
Write-Host "  Output: $OutputDir"
