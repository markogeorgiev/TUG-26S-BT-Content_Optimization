[CmdletBinding()]
param(
    [string]$BaseCategory = 'Category:Pets',
    [string]$OutputDir = '',
    [int]$RequestDelayMs = 3000,
    [int]$RequestDelayJitterMs = 5000,
    [int]$MaxRetries = 3,
    [switch]$IncludeAllNamespaces,
    [switch]$RetryFailedPages
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $OutputDir = Join-Path $repoRoot 'output\wikipedia-pets'
}

$wikiBaseUrl = 'https://en.wikipedia.org'
$apiUrl = "$wikiBaseUrl/w/api.php"
$userAgent = 'TUG-26S-BT-Content_Optimization/1.0 (PowerShell Wikipedia category scraper)'
$random = [System.Random]::new()
$discoveryMemberCheckpointInterval = 25

function Get-RandomizedDelayMilliseconds {
    param(
        [Parameter(Mandatory = $true)]
        [int]$BaseDelayMs,
        [Parameter(Mandatory = $true)]
        [int]$JitterMs
    )

    if ($BaseDelayMs -lt 0) {
        $BaseDelayMs = 0
    }

    if ($JitterMs -lt 0) {
        $JitterMs = 0
    }

    if ($JitterMs -eq 0) {
        return $BaseDelayMs
    }

    return $BaseDelayMs + $random.Next(0, $JitterMs + 1)
}

function Start-PoliteDelay {
    $delayMs = Get-RandomizedDelayMilliseconds -BaseDelayMs $RequestDelayMs -JitterMs $RequestDelayJitterMs
    if ($delayMs -gt 0) {
        Start-Sleep -Milliseconds $delayMs
    }
}

function Get-RetryDelaySeconds {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Attempt,
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    $response = Get-ExceptionResponse -ErrorRecord $ErrorRecord
    if ($response) {
        try {
            $statusCode = [int]$response.StatusCode
        }
        catch {
            $statusCode = $null
        }

        if ($statusCode -eq 429) {
            $retryAfterSeconds = $null

            try {
                $retryAfterHeader = $response.Headers['Retry-After']
                if ($retryAfterHeader) {
                    [int]::TryParse($retryAfterHeader, [ref]$retryAfterSeconds) | Out-Null
                }
            }
            catch {
            }

            if ($retryAfterSeconds -and $retryAfterSeconds -gt 0) {
                $jitterMs = Get-RandomizedDelayMilliseconds -BaseDelayMs 1000 -JitterMs 4000
                return $retryAfterSeconds + [Math]::Ceiling($jitterMs / 1000.0)
            }

            $baseDelay = [Math]::Min(120, 20 * $Attempt)
            $jitterMs = Get-RandomizedDelayMilliseconds -BaseDelayMs 2000 -JitterMs 6000
            return $baseDelay + [Math]::Ceiling($jitterMs / 1000.0)
        }
    }

    $baseSeconds = [Math]::Min(30, [Math]::Pow(2, $Attempt))
    $jitterMs = Get-RandomizedDelayMilliseconds -BaseDelayMs 500 -JitterMs 2500
    return $baseSeconds + [Math]::Ceiling($jitterMs / 1000.0)
}

function Get-ExceptionResponse {
    param(
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    if (-not $ErrorRecord.Exception) {
        return $null
    }

    if ($ErrorRecord.Exception.PSObject.Properties.Name -contains 'Response') {
        return $ErrorRecord.Exception.Response
    }

    return $null
}

function Test-ShouldRetryError {
    param(
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    $statusCode = Get-HttpStatusCodeFromErrorRecord -ErrorRecord $ErrorRecord
    if ($statusCode -in @(404, 410)) {
        return $false
    }

    return $true
}

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Operation,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $attempt = 0

    while ($attempt -lt $MaxRetries) {
        $attempt++

        try {
            return & $Operation
        }
        catch {
            if (-not (Test-ShouldRetryError -ErrorRecord $_)) {
                $statusCode = Get-HttpStatusCodeFromErrorRecord -ErrorRecord $_
                throw "Failed to $Description. Non-retryable HTTP status $statusCode. $($_.Exception.Message)"
            }

            if ($attempt -ge $MaxRetries) {
                throw "Failed to $Description after $attempt attempt(s). $($_.Exception.Message)"
            }

            $delaySeconds = Get-RetryDelaySeconds -Attempt $attempt -ErrorRecord $_
            Write-Warning "$Description failed on attempt $attempt. Retrying in $delaySeconds second(s). $($_.Exception.Message)"
            Start-Sleep -Seconds $delaySeconds
        }
    }
}

function Invoke-WikipediaApi {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Parameters,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $requestParameters = @{
        format        = 'json'
        formatversion = '2'
    }

    foreach ($key in $Parameters.Keys) {
        $requestParameters[$key] = $Parameters[$key]
    }

    $result = Invoke-WithRetry -Description $Description -Operation {
        Invoke-RestMethod `
            -Method Get `
            -Uri $apiUrl `
            -Headers @{ 'User-Agent' = $userAgent } `
            -Body $requestParameters
    }

    Start-PoliteDelay

    return $result
}

function Invoke-WikipediaPageRequest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    $encodedTitle = [Uri]::EscapeDataString($Title.Replace(' ', '_'))
    $url = "$wikiBaseUrl/wiki/$encodedTitle"

    $response = Invoke-WithRetry -Description "fetch page HTML for '$Title'" -Operation {
        Invoke-WebRequest `
            -Method Get `
            -Uri $url `
            -Headers @{ 'User-Agent' = $userAgent } `
            -MaximumRedirection 5 `
            -UseBasicParsing
    }

    Start-PoliteDelay

    return $response
}

function Get-PageInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    $response = Invoke-WikipediaApi `
        -Description "resolve page info for '$Title'" `
        -Parameters @{
            action    = 'query'
            redirects = '1'
            titles    = $Title
        }

    if (-not $response.query.pages -or $response.query.pages.Count -eq 0) {
        throw "Could not resolve page info for '$Title'."
    }

    $page = $response.query.pages[0]
    $kind = if ($page.ns -eq 14) { 'category' } else { 'page' }

    return [pscustomobject]@{
        PageId = [int]$page.pageid
        Title  = [string]$page.title
        Kind   = $kind
    }
}

function Normalize-WikiTitleFromHref {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Href
    )

    $prefixes = @(
        '/wiki/',
        'https://en.wikipedia.org/wiki/'
    )

    $rawTitle = $null

    foreach ($prefix in $prefixes) {
        if ($Href.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rawTitle = $Href.Substring($prefix.Length)
            break
        }
    }

    if (-not $rawTitle) {
        return $null
    }

    $rawTitle = $rawTitle.Split('#')[0]
    $rawTitle = $rawTitle.Split('?')[0]

    if ([string]::IsNullOrWhiteSpace($rawTitle)) {
        return $null
    }

    return [Uri]::UnescapeDataString($rawTitle).Replace('_', ' ').Trim()
}

function Get-SafeFileStem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    $invalidCharacters = [IO.Path]::GetInvalidFileNameChars()
    $safeTitle = -join ($Title.ToCharArray() | ForEach-Object {
        if ($invalidCharacters -contains $_) {
            '-'
        }
        else {
            $_
        }
    })

    $safeTitle = $safeTitle -replace '\s+', '_'
    $safeTitle = $safeTitle.Trim('_', '.')

    if ($safeTitle.Length -gt 80) {
        $safeTitle = $safeTitle.Substring(0, 80).TrimEnd('_')
    }

    $sha1 = [System.Security.Cryptography.SHA1]::Create()
    try {
        $hashBytes = $sha1.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Title))
    }
    finally {
        $sha1.Dispose()
    }

    $hash = ([System.BitConverter]::ToString($hashBytes)).Replace('-', '').ToLowerInvariant().Substring(0, 10)
    return "$safeTitle`_$hash"
}

function Get-MatchedBlock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Html,
        [Parameter(Mandatory = $true)]
        [string[]]$StartMarkers,
        [Parameter(Mandatory = $true)]
        [string[]]$EndMarkers
    )

    $startIndex = -1

    foreach ($startMarker in $StartMarkers) {
        $match = [regex]::Match($Html, $startMarker, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($match.Success) {
            $startIndex = $match.Index
            break
        }
    }

    if ($startIndex -lt 0) {
        return $Html
    }

    $segment = $Html.Substring($startIndex)
    $endIndexes = New-Object System.Collections.Generic.List[int]

    foreach ($endMarker in $EndMarkers) {
        $match = [regex]::Match($segment, $endMarker, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($match.Success -and $match.Index -gt 0) {
            $endIndexes.Add($match.Index)
        }
    }

    if ($endIndexes.Count -gt 0) {
        $endIndex = ($endIndexes | Measure-Object -Minimum).Minimum
        return $segment.Substring(0, $endIndex)
    }

    return $segment
}

function Get-ContentHtmlFragment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Html
    )

    return Get-MatchedBlock `
        -Html $Html `
        -StartMarkers @(
            '<div id="mw-content-text"[^>]*>',
            '<div class="mw-category-generated"[^>]*>'
        ) `
        -EndMarkers @(
            '<div class="printfooter"',
            '<div id="catlinks"',
            '<nav id="mw-navigation"',
            '<div id="mw-navigation"',
            '<footer id="footer"'
        )
}

function Remove-HtmlPattern {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputHtml,
        [Parameter(Mandatory = $true)]
        [string]$Pattern
    )

    return [regex]::Replace(
        $InputHtml,
        $Pattern,
        '',
        [System.Text.RegularExpressions.RegexOptions]::IgnoreCase -bor
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
}

function Convert-HtmlToPlainText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HtmlFragment
    )

    $cleanHtml = $HtmlFragment

    foreach ($pattern in @(
        '<!--.*?-->',
        '<script\b.*?</script>',
        '<style\b.*?</style>',
        '<noscript\b.*?</noscript>',
        '<figure\b.*?</figure>',
        '<table\b[^>]*class="[^"]*(?:infobox|navbox|vertical-navbox|sidebar|metadata|plainlinks|toccolours)[^"]*"[^>]*>.*?</table>',
        '<div\b[^>]*class="[^"]*(?:shortdescription|navbox|metadata|authority-control|reflist|mw-editsection|toc|thumb|sistersitebox|portalbox|ambox|plainlinks)[^"]*"[^>]*>.*?</div>',
        '<span\b[^>]*class="[^"]*(?:mw-editsection|reference-text|error)[^"]*"[^>]*>.*?</span>',
        '<sup\b[^>]*class="[^"]*reference[^"]*"[^>]*>.*?</sup>'
    )) {
        $cleanHtml = Remove-HtmlPattern -InputHtml $cleanHtml -Pattern $pattern
    }

    $cleanHtml = [regex]::Replace($cleanHtml, '<br\s*/?>', "`n", 'IgnoreCase')
    $cleanHtml = [regex]::Replace($cleanHtml, '<li\b[^>]*>', "`n- ", 'IgnoreCase')
    $cleanHtml = [regex]::Replace($cleanHtml, '</li>', "`n", 'IgnoreCase')
    $cleanHtml = [regex]::Replace($cleanHtml, '<(?:p|div|section|tr|ul|ol|dl|dt|dd|h[1-6])\b[^>]*>', "`n", 'IgnoreCase')
    $cleanHtml = [regex]::Replace($cleanHtml, '</(?:p|div|section|tr|ul|ol|dl|dt|dd|h[1-6])>', "`n", 'IgnoreCase')
    $cleanHtml = [regex]::Replace($cleanHtml, '<[^>]+>', ' ')

    $decoded = [System.Net.WebUtility]::HtmlDecode($cleanHtml)
    $decoded = $decoded.Replace([char]0x00A0, ' ')
    $decoded = $decoded -replace "`r", ''
    $decoded = [regex]::Replace($decoded, "[ \t]+`n", "`n")
    $decoded = [regex]::Replace($decoded, "`n[ \t]+", "`n")
    $decoded = [regex]::Replace($decoded, '[ \t]{2,}', ' ')
    $decoded = [regex]::Replace($decoded, "`n{3,}", "`n`n")

    return $decoded.Trim()
}

function Get-InternalLinksFromHtml {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HtmlFragment
    )

    $results = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::Ordinal)
    $matches = [regex]::Matches($HtmlFragment, '<a\b[^>]*\bhref="(?<href>[^"]+)"[^>]*>', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)

    foreach ($match in $matches) {
        $title = Normalize-WikiTitleFromHref -Href $match.Groups['href'].Value
        if ($title) {
            [void]$results.Add($title)
        }
    }

    return @($results)
}

function Get-CanonicalTitleFromUri {
    param(
        [Parameter(Mandatory = $true)]
        [Uri]$Uri
    )

    return Normalize-WikiTitleFromHref -Href $Uri.AbsoluteUri
}

function New-NodeRecord {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [int]$PageId,
        [Parameter(Mandatory = $true)]
        [string]$Kind,
        [string]$RequestedTitle = '',
        [string[]]$DiscoveredFrom = @()
    )

    if ([string]::IsNullOrWhiteSpace($RequestedTitle)) {
        $RequestedTitle = $Title
    }

    return [ordered]@{
        title           = $Title
        page_id         = $PageId
        kind            = $Kind
        url             = $null
        text_file       = $null
        requested_title = $RequestedTitle
        canonical_title = $null
        discovered_from = @($DiscoveredFrom)
        links           = @()
        fetch_status    = 'pending'
        failure_count   = 0
        last_error      = $null
        last_http_status = $null
        last_attempt_utc = $null
    }
}

function Save-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Data
    )

    $tempPath = "$Path.tmp"
    $Data | ConvertTo-Json -Depth 8 | Set-Content -Path $tempPath -Encoding UTF8
    Move-Item -Path $tempPath -Destination $Path -Force
}

function Load-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $raw = Get-Content -LiteralPath $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }

    return $raw | ConvertFrom-Json
}

function Get-OptionalPropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$InputObject,
        [Parameter(Mandatory = $true)]
        [string]$PropertyName,
        $DefaultValue = $null
    )

    if (-not $InputObject) {
        return $DefaultValue
    }

    if ($InputObject.PSObject.Properties.Name -contains $PropertyName) {
        return $InputObject.$PropertyName
    }

    return $DefaultValue
}

function Get-NodeProgressFilePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StateDirectory,
        [Parameter(Mandatory = $true)]
        [string]$Title
    )

    $fileStem = Get-SafeFileStem -Title $Title
    return Join-Path $StateDirectory "$fileStem.json"
}

function Save-CrawlInventory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$RequestedRootCategory,
        [Parameter(Mandatory = $true)]
        [string]$RootCategory,
        [Parameter(Mandatory = $true)]
        [bool]$IncludeAllNamespacesValue,
        [Parameter(Mandatory = $true)]
        [hashtable]$Nodes
    )

    $inventoryNodes = foreach ($title in ($Nodes.Keys | Sort-Object)) {
        $node = $Nodes[$title]
        [pscustomobject]@{
            title           = $node.title
            page_id         = $node.page_id
            kind            = $node.kind
            requested_title = $node.requested_title
            discovered_from = @($node.discovered_from | Sort-Object)
        }
    }

    $inventory = [pscustomobject]@{
        schema_version          = 1
        requested_root_category = $RequestedRootCategory
        root_category           = $RootCategory
        include_all_namespaces  = $IncludeAllNamespacesValue
        discovered_at_utc       = (Get-Date).ToUniversalTime().ToString('o')
        node_count              = $Nodes.Count
        nodes                   = @($inventoryNodes)
    }

    Save-JsonFile -Path $Path -Data $inventory
}

function Save-DiscoveryState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$RequestedRootCategory,
        [Parameter(Mandatory = $true)]
        [string]$RootCategory,
        [Parameter(Mandatory = $true)]
        [bool]$IncludeAllNamespacesValue,
        [Parameter(Mandatory = $true)]
        [hashtable]$Nodes,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$SeenCategories,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$PendingCategories,
        [string]$CurrentCategory,
        [string]$CurrentCategoryContinueToken
    )

    $discoveryNodes = foreach ($title in ($Nodes.Keys | Sort-Object)) {
        $node = $Nodes[$title]
        [pscustomobject]@{
            title           = $node.title
            page_id         = $node.page_id
            kind            = $node.kind
            requested_title = $node.requested_title
            discovered_from = @($node.discovered_from | Sort-Object)
        }
    }

    $state = [pscustomobject]@{
        schema_version          = 1
        requested_root_category = $RequestedRootCategory
        root_category           = $RootCategory
        include_all_namespaces  = $IncludeAllNamespacesValue
        saved_at_utc            = (Get-Date).ToUniversalTime().ToString('o')
        node_count              = $Nodes.Count
        seen_categories         = @($SeenCategories)
        pending_categories      = @($PendingCategories)
        current_category        = $CurrentCategory
        current_category_continue_token = $CurrentCategoryContinueToken
        nodes                   = @($discoveryNodes)
    }

    Save-JsonFile -Path $Path -Data $state
}

function Restore-NodesFromInventory {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Inventory
    )

    $restoredNodes = @{}

    foreach ($node in @($Inventory.nodes)) {
        $requestedTitle = if ($node.requested_title) { [string]$node.requested_title } else { [string]$node.title }
        $restoredNodes[[string]$node.title] = New-NodeRecord `
            -Title ([string]$node.title) `
            -PageId ([int]$node.page_id) `
            -Kind ([string]$node.kind) `
            -RequestedTitle $requestedTitle `
            -DiscoveredFrom @($node.discovered_from | ForEach-Object { [string]$_ })
    }

    return $restoredNodes
}

function Invoke-CategoryMembersTraversal {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CategoryTitle,
        [string]$StartContinueToken,
        [Parameter(Mandatory = $true)]
        [scriptblock]$OnMember,
        [scriptblock]$OnCheckpoint
    )

    $continueToken = if ([string]::IsNullOrWhiteSpace($StartContinueToken)) { $null } else { $StartContinueToken }

    do {
        if ($OnCheckpoint) {
            & $OnCheckpoint $continueToken 'before_page'
        }

        $parameters = @{
            action  = 'query'
            list    = 'categorymembers'
            cmtitle = $CategoryTitle
            cmtype  = 'page|subcat'
            cmprop  = 'ids|title|type'
            cmlimit = 'max'
        }

        if ($continueToken) {
            $parameters.cmcontinue = $continueToken
        }

        $response = Invoke-WikipediaApi `
            -Description "list category members for '$CategoryTitle'" `
            -Parameters $parameters

        $processedMembersInPage = 0
        foreach ($member in @($response.query.categorymembers)) {
            if (-not $IncludeAllNamespaces -and $member.ns -notin @(0, 14)) {
                continue
            }

            $kind = if ($member.type -eq 'subcat' -or $member.ns -eq 14) { 'category' } else { 'page' }
            & $OnMember ([pscustomobject]@{
                PageId = [int]$member.pageid
                Title  = [string]$member.title
                Kind   = $kind
            })

            $processedMembersInPage++
            if ($OnCheckpoint -and ($processedMembersInPage % $discoveryMemberCheckpointInterval) -eq 0) {
                & $OnCheckpoint $continueToken 'within_page'
            }
        }

        $nextContinueToken = $null
        if ($response.PSObject.Properties.Name -contains 'continue' -and $response.continue.cmcontinue) {
            $nextContinueToken = [string]$response.continue.cmcontinue
        }

        if ($OnCheckpoint) {
            & $OnCheckpoint $nextContinueToken 'after_page'
        }

        $continueToken = $nextContinueToken
    }
    while ($continueToken)
}

function Test-ProgressConfigurationMatches {
    param(
        [Parameter(Mandatory = $true)]
        [object]$SavedState,
        [Parameter(Mandatory = $true)]
        [string]$RequestedBaseCategory,
        [Parameter(Mandatory = $true)]
        [bool]$IncludeAllNamespacesValue
    )

    $savedRequestedRootValue = Get-OptionalPropertyValue -InputObject $SavedState -PropertyName 'requested_root_category' -DefaultValue ''
    $savedRootCategoryValue = Get-OptionalPropertyValue -InputObject $SavedState -PropertyName 'root_category' -DefaultValue ''
    $savedNamespacesValue = Get-OptionalPropertyValue -InputObject $SavedState -PropertyName 'include_all_namespaces' -DefaultValue $false

    $savedRequestedRoot = if ($savedRequestedRootValue) { [string]$savedRequestedRootValue } else { '' }
    $savedRootCategory = if ($savedRootCategoryValue) { [string]$savedRootCategoryValue } else { '' }
    $savedNamespacesFlag = [bool]$savedNamespacesValue

    $rootMatches = $false
    if ($savedRequestedRoot.Equals($RequestedBaseCategory, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rootMatches = $true
    }
    elseif ($savedRootCategory.Equals($RequestedBaseCategory, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rootMatches = $true
    }

    return ($rootMatches -and ($savedNamespacesFlag -eq $IncludeAllNamespacesValue))
}

function Save-NodeProgress {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StateDirectory,
        [Parameter(Mandatory = $true)]
        [hashtable]$Node
    )

    $progressPath = Get-NodeProgressFilePath -StateDirectory $StateDirectory -Title $Node.title
    $progressRecord = [pscustomobject]@{
        schema_version   = 1
        title            = $Node.title
        fetch_status     = $Node.fetch_status
        url              = $Node.url
        text_file        = $Node.text_file
        canonical_title  = $Node.canonical_title
        links            = @($Node.links)
        failure_count    = $Node.failure_count
        last_error       = $Node.last_error
        last_http_status = $Node.last_http_status
        last_attempt_utc = $Node.last_attempt_utc
    }

    Save-JsonFile -Path $progressPath -Data $progressRecord
}

function Apply-SavedProgress {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Nodes,
        [Parameter(Mandatory = $true)]
        [string]$StateDirectory
    )

    foreach ($title in @($Nodes.Keys)) {
        $progressPath = Get-NodeProgressFilePath -StateDirectory $StateDirectory -Title $title
        $progress = Load-JsonFile -Path $progressPath

        if (-not $progress) {
            continue
        }

        $node = $Nodes[$title]
        $urlValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'url'
        $textFileValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'text_file'
        $canonicalTitleValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'canonical_title'
        $linksValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'links' -DefaultValue @()
        $fetchStatusValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'fetch_status' -DefaultValue 'pending'
        $failureCountValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'failure_count' -DefaultValue 0
        $lastErrorValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'last_error'
        $lastHttpStatusValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'last_http_status'
        $lastAttemptUtcValue = Get-OptionalPropertyValue -InputObject $progress -PropertyName 'last_attempt_utc'

        $node.url = if ($urlValue) { [string]$urlValue } else { $null }
        $node.text_file = if ($textFileValue) { [string]$textFileValue } else { $null }
        $node.canonical_title = if ($canonicalTitleValue) { [string]$canonicalTitleValue } else { $null }
        $node.links = @($linksValue | ForEach-Object { [string]$_ })
        $node.fetch_status = if ($fetchStatusValue) { [string]$fetchStatusValue } else { 'pending' }
        $node.failure_count = if ($null -ne $failureCountValue) { [int]$failureCountValue } else { 0 }
        $node.last_error = if ($lastErrorValue) { [string]$lastErrorValue } else { $null }
        $node.last_http_status = if ($null -ne $lastHttpStatusValue) { [int]$lastHttpStatusValue } else { $null }
        $node.last_attempt_utc = if ($lastAttemptUtcValue) { [string]$lastAttemptUtcValue } else { $null }
    }
}

function Read-ExistingTextFileMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $lines = Get-Content -LiteralPath $Path -TotalCount 8
    if (-not $lines -or $lines.Count -eq 0) {
        return $null
    }

    $metadata = @{
        title           = $null
        canonical_title = $null
        kind            = $null
        url             = $null
    }

    foreach ($line in $lines) {
        if ($line -match '^Title:\s*(.+)$') {
            $metadata.title = $Matches[1].Trim()
            continue
        }

        if ($line -match '^CanonicalTitle:\s*(.+)$') {
            $metadata.canonical_title = $Matches[1].Trim()
            continue
        }

        if ($line -match '^Kind:\s*(.+)$') {
            $metadata.kind = $Matches[1].Trim()
            continue
        }

        if ($line -match '^URL:\s*(.+)$') {
            $metadata.url = $Matches[1].Trim()
            continue
        }
    }

    return [pscustomobject]$metadata
}

function Bootstrap-NodesFromExistingTextFiles {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Nodes,
        [Parameter(Mandatory = $true)]
        [string]$OutputDirectory,
        [Parameter(Mandatory = $true)]
        [string]$TextsDirectory,
        [Parameter(Mandatory = $true)]
        [string]$StateDirectory
    )

    $bootstrappedCount = 0

    foreach ($title in @($Nodes.Keys)) {
        $node = $Nodes[$title]

        if ($node.fetch_status -eq 'completed') {
            continue
        }

        $fileStem = Get-SafeFileStem -Title $title
        $fileName = "$fileStem.txt"
        $relativeTextPath = "texts/$fileName"
        $absoluteTextPath = Join-Path $TextsDirectory $fileName

        if (-not (Test-Path -LiteralPath $absoluteTextPath)) {
            continue
        }

        $metadata = Read-ExistingTextFileMetadata -Path $absoluteTextPath

        $node.text_file = $relativeTextPath
        if ($metadata -and $metadata.url) {
            $node.url = [string]$metadata.url
        }
        if ($metadata -and $metadata.canonical_title) {
            $node.canonical_title = [string]$metadata.canonical_title
        }
        if ($metadata -and $metadata.kind) {
            $node.kind = [string]$metadata.kind
        }

        $node.links = @()
        $node.fetch_status = 'completed'
        $node.last_error = $null
        $node.last_http_status = $null

        $lastWriteUtc = (Get-Item -LiteralPath $absoluteTextPath).LastWriteTimeUtc
        $node.last_attempt_utc = $lastWriteUtc.ToString('o')

        Save-NodeProgress -StateDirectory $StateDirectory -Node $node
        $bootstrappedCount++
    }

    return $bootstrappedCount
}

function Get-HttpStatusCodeFromErrorRecord {
    param(
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    $response = Get-ExceptionResponse -ErrorRecord $ErrorRecord
    if (-not $response) {
        return $null
    }

    try {
        return [int]$response.StatusCode
    }
    catch {
        return $null
    }
}

function Get-NodeFetchCounts {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Nodes
    )

    $completed = 0
    $failed = 0
    $pending = 0

    foreach ($node in @($Nodes.Values)) {
        switch ($node.fetch_status) {
            'completed' { $completed++ }
            'failed' { $failed++ }
            default { $pending++ }
        }
    }

    return [pscustomobject]@{
        completed = $completed
        failed    = $failed
        pending   = $pending
    }
}

$textsDir = Join-Path $OutputDir 'texts'
$progressDir = Join-Path $OutputDir 'progress'
$pageStateDir = Join-Path $progressDir 'page-state'
$inventoryPath = Join-Path $progressDir 'inventory.json'
$discoveryStatePath = Join-Path $progressDir 'discovery_state.json'

New-Item -ItemType Directory -Path $textsDir -Force | Out-Null
New-Item -ItemType Directory -Path $progressDir -Force | Out-Null
New-Item -ItemType Directory -Path $pageStateDir -Force | Out-Null

$nodes = @{}
$rootCategory = $null
$inventory = Load-JsonFile -Path $inventoryPath
$discoveryState = if (-not $inventory) { Load-JsonFile -Path $discoveryStatePath } else { $null }

if ($inventory) {
    if (-not (Test-ProgressConfigurationMatches -SavedState $inventory -RequestedBaseCategory $BaseCategory -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces))) {
        throw "Existing progress in '$progressDir' does not match this crawl configuration. Use the same BaseCategory and namespace mode, or choose a different OutputDir."
    }

    Write-Host "Loading saved crawl inventory from $inventoryPath ..."
    $nodes = Restore-NodesFromInventory -Inventory $inventory
    Apply-SavedProgress -Nodes $nodes -StateDirectory $pageStateDir
    $rootCategory = [string](Get-OptionalPropertyValue -InputObject $inventory -PropertyName 'root_category' -DefaultValue $BaseCategory)
}
else {
    $categoryQueue = New-Object 'System.Collections.Generic.Queue[string]'
    $seenCategories = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::Ordinal)
    $processedCategoryCount = 0
    $resumeCurrentCategory = $null
    $resumeCurrentCategoryContinueToken = $null

    if ($discoveryState) {
        if (-not (Test-ProgressConfigurationMatches -SavedState $discoveryState -RequestedBaseCategory $BaseCategory -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces))) {
            throw "Existing discovery progress in '$progressDir' does not match this crawl configuration. Use the same BaseCategory and namespace mode, or choose a different OutputDir."
        }

        Write-Host "Resuming category discovery from $discoveryStatePath ..."
        $nodes = Restore-NodesFromInventory -Inventory $discoveryState
        $rootCategory = [string](Get-OptionalPropertyValue -InputObject $discoveryState -PropertyName 'root_category' -DefaultValue $BaseCategory)

        foreach ($seenCategory in @(Get-OptionalPropertyValue -InputObject $discoveryState -PropertyName 'seen_categories' -DefaultValue @())) {
            [void]$seenCategories.Add([string]$seenCategory)
        }

        $savedCurrentCategory = Get-OptionalPropertyValue -InputObject $discoveryState -PropertyName 'current_category'
        $savedCurrentCategoryContinueToken = Get-OptionalPropertyValue -InputObject $discoveryState -PropertyName 'current_category_continue_token'

        if ($savedCurrentCategory) {
            $resumeCurrentCategory = [string]$savedCurrentCategory
            $resumeCurrentCategoryContinueToken = if ($savedCurrentCategoryContinueToken) { [string]$savedCurrentCategoryContinueToken } else { $null }
            $categoryQueue.Enqueue($resumeCurrentCategory)
        }

        foreach ($pendingCategory in @(Get-OptionalPropertyValue -InputObject $discoveryState -PropertyName 'pending_categories' -DefaultValue @())) {
            $pendingCategoryText = [string]$pendingCategory
            if ($resumeCurrentCategory -and $pendingCategoryText.Equals($resumeCurrentCategory, [System.StringComparison]::OrdinalIgnoreCase)) {
                continue
            }
            $categoryQueue.Enqueue($pendingCategoryText)
        }
    }
    else {
        Write-Host "Resolving root category $BaseCategory ..."
        $rootInfo = Get-PageInfo -Title $BaseCategory
        $rootCategory = $rootInfo.Title

        $nodes[$rootInfo.Title] = New-NodeRecord `
            -Title $rootInfo.Title `
            -PageId $rootInfo.PageId `
            -Kind $rootInfo.Kind `
            -RequestedTitle $rootInfo.Title `
            -DiscoveredFrom @()

        $categoryQueue.Enqueue($rootInfo.Title)

        Save-DiscoveryState `
            -Path $discoveryStatePath `
            -RequestedRootCategory $BaseCategory `
            -RootCategory $rootCategory `
            -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces) `
            -Nodes $nodes `
            -SeenCategories @() `
            -PendingCategories @($categoryQueue.ToArray()) `
            -CurrentCategory $null `
            -CurrentCategoryContinueToken $null
    }

    Write-Host 'Discovering descendant pages and subcategories ...'

    while ($categoryQueue.Count -gt 0) {
        $currentCategory = $categoryQueue.Dequeue()
        $isResumingCurrentCategory = $resumeCurrentCategory -and $currentCategory.Equals($resumeCurrentCategory, [System.StringComparison]::OrdinalIgnoreCase)

        if ($seenCategories.Contains($currentCategory) -and -not $isResumingCurrentCategory) {
            continue
        }

        $processedCategoryCount++
        if ($isResumingCurrentCategory -and $resumeCurrentCategoryContinueToken) {
            Write-Host "  Resuming category: $currentCategory"
        }
        else {
            Write-Host "  Crawling category: $currentCategory"
        }

        $memberHandler = {
            param($member)

            if (-not $nodes.ContainsKey($member.Title)) {
                $nodes[$member.Title] = New-NodeRecord `
                    -Title $member.Title `
                    -PageId $member.PageId `
                    -Kind $member.Kind `
                    -RequestedTitle $member.Title `
                    -DiscoveredFrom @($currentCategory)
            }
            else {
                $existingSources = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::Ordinal)
                foreach ($source in @($nodes[$member.Title].discovered_from)) {
                    [void]$existingSources.Add($source)
                }
                [void]$existingSources.Add($currentCategory)
                $nodes[$member.Title].discovered_from = @($existingSources)
            }

            if ($member.Kind -eq 'category' -and -not $seenCategories.Contains($member.Title)) {
                $alreadyQueued = $false
                foreach ($queuedCategory in $categoryQueue.ToArray()) {
                    if ($queuedCategory.Equals($member.Title, [System.StringComparison]::OrdinalIgnoreCase)) {
                        $alreadyQueued = $true
                        break
                    }
                }

                if (-not $alreadyQueued -and -not ($resumeCurrentCategory -and $resumeCurrentCategory.Equals($member.Title, [System.StringComparison]::OrdinalIgnoreCase))) {
                    $categoryQueue.Enqueue($member.Title)
                }
            }
        }

        $checkpointWriter = {
            param($checkpointContinueToken, $checkpointPhase)

            Save-DiscoveryState `
                -Path $discoveryStatePath `
                -RequestedRootCategory $BaseCategory `
                -RootCategory $rootCategory `
                -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces) `
                -Nodes $nodes `
                -SeenCategories @($seenCategories) `
                -PendingCategories @($categoryQueue.ToArray()) `
                -CurrentCategory $currentCategory `
                -CurrentCategoryContinueToken $checkpointContinueToken
        }

        Invoke-CategoryMembersTraversal `
            -CategoryTitle $currentCategory `
            -StartContinueToken $resumeCurrentCategoryContinueToken `
            -OnMember $memberHandler `
            -OnCheckpoint $checkpointWriter

        [void]$seenCategories.Add($currentCategory)
        $resumeCurrentCategory = $null
        $resumeCurrentCategoryContinueToken = $null

        if (($processedCategoryCount % 5) -eq 0 -or $categoryQueue.Count -eq 0) {
            Save-DiscoveryState `
                -Path $discoveryStatePath `
                -RequestedRootCategory $BaseCategory `
                -RootCategory $rootCategory `
                -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces) `
                -Nodes $nodes `
                -SeenCategories @($seenCategories) `
                -PendingCategories @($categoryQueue.ToArray()) `
                -CurrentCategory $null `
                -CurrentCategoryContinueToken $null
        }
    }

    Save-CrawlInventory `
        -Path $inventoryPath `
        -RequestedRootCategory $BaseCategory `
        -RootCategory $rootCategory `
        -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces) `
        -Nodes $nodes

    Save-DiscoveryState `
        -Path $discoveryStatePath `
        -RequestedRootCategory $BaseCategory `
        -RootCategory $rootCategory `
        -IncludeAllNamespacesValue ([bool]$IncludeAllNamespaces) `
        -Nodes $nodes `
        -SeenCategories @($seenCategories) `
        -PendingCategories @() `
        -CurrentCategory $null `
        -CurrentCategoryContinueToken $null
}

$bootstrappedCount = Bootstrap-NodesFromExistingTextFiles `
    -Nodes $nodes `
    -OutputDirectory $OutputDir `
    -TextsDirectory $textsDir `
    -StateDirectory $pageStateDir

if ($bootstrappedCount -gt 0) {
    Write-Host "Bootstrapped $bootstrappedCount completed page(s) from existing text files."
}

Write-Host ("Discovered {0} total nodes ({1} categories, {2} pages)." -f `
    $nodes.Count, `
    (@($nodes.Values | Where-Object { $_.kind -eq 'category' })).Count, `
    (@($nodes.Values | Where-Object { $_.kind -eq 'page' })).Count)

$counts = Get-NodeFetchCounts -Nodes $nodes
Write-Host ("Resume state: {0} completed, {1} failed, {2} pending." -f $counts.completed, $counts.failed, $counts.pending)

$titleSet = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::Ordinal)
foreach ($title in $nodes.Keys) {
    [void]$titleSet.Add($title)
}

$orderedTitles = @($nodes.Keys | Sort-Object)

for ($pageIndex = 0; $pageIndex -lt $orderedTitles.Count; $pageIndex++) {
    $title = $orderedTitles[$pageIndex]
    $node = $nodes[$title]

    $textFileExists = $false
    if ($node.text_file) {
        $textFileExists = Test-Path -LiteralPath (Join-Path $OutputDir $node.text_file)
    }

    if ($node.fetch_status -eq 'completed' -and $textFileExists) {
        continue
    }

    if ($node.fetch_status -eq 'failed' -and -not $RetryFailedPages) {
        continue
    }

    $actionLabel = if ($node.fetch_status -eq 'failed') { 'Retrying' } else { 'Fetching' }
    Write-Host ("  [{0}/{1}] {2} {3}" -f ($pageIndex + 1), $orderedTitles.Count, $actionLabel, $title)

    try {
        $response = Invoke-WikipediaPageRequest -Title $title
        $contentHtml = Get-ContentHtmlFragment -Html $response.Content
        $plainText = Convert-HtmlToPlainText -HtmlFragment $contentHtml
        $pageLinks = Get-InternalLinksFromHtml -HtmlFragment $contentHtml
        $canonicalTitle = Get-CanonicalTitleFromUri -Uri $response.BaseResponse.ResponseUri

        $fileStem = Get-SafeFileStem -Title $title
        $fileName = "$fileStem.txt"
        $filePath = Join-Path $textsDir $fileName

        $fileBody = @(
            "Title: $title"
            "CanonicalTitle: $canonicalTitle"
            "Kind: $($node.kind)"
            "URL: $($response.BaseResponse.ResponseUri.AbsoluteUri)"
            ''
            $plainText
        ) -join "`r`n"

        Set-Content -Path $filePath -Value $fileBody -Encoding UTF8

        $node.url = $response.BaseResponse.ResponseUri.AbsoluteUri
        $node.text_file = "texts/$fileName"
        $node.canonical_title = $canonicalTitle
        $node.links = @($pageLinks)
        $node.fetch_status = 'completed'
        $node.last_error = $null
        $node.last_http_status = $null
        $node.last_attempt_utc = (Get-Date).ToUniversalTime().ToString('o')

        Save-NodeProgress -StateDirectory $pageStateDir -Node $node
    }
    catch {
        $node.fetch_status = 'failed'
        $node.failure_count = [int]$node.failure_count + 1
        $node.last_error = $_.Exception.Message
        $node.last_http_status = Get-HttpStatusCodeFromErrorRecord -ErrorRecord $_
        $node.last_attempt_utc = (Get-Date).ToUniversalTime().ToString('o')

        Save-NodeProgress -StateDirectory $pageStateDir -Node $node
        Write-Warning "Skipping '$title' after failure. Progress has been saved and the crawl will continue. $($node.last_error)"
        continue
    }
}

$edges = New-Object System.Collections.Generic.List[object]
$edgeKeys = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::Ordinal)

foreach ($title in $orderedTitles) {
    $node = $nodes[$title]

    foreach ($targetTitle in @($node.links)) {
        if (-not $titleSet.Contains($targetTitle)) {
            continue
        }

        if ($targetTitle -eq $title) {
            continue
        }

        $edgeKey = "$title`t$targetTitle"
        if (-not $edgeKeys.Add($edgeKey)) {
            continue
        }

        $targetNode = $nodes[$targetTitle]
        $edges.Add([pscustomobject]@{
            source_title   = $title
            source_kind    = $node.kind
            source_page_id = $node.page_id
            source_url     = $node.url
            target_title   = $targetTitle
            target_kind    = $targetNode.kind
            target_page_id = $targetNode.page_id
            target_url     = $targetNode.url
        })
    }
}

$pageIndex = @()
foreach ($title in $orderedTitles) {
    $node = $nodes[$title]
    $pageIndex += [pscustomobject]@{
        title           = $node.title
        canonical_title = $node.canonical_title
        page_id         = $node.page_id
        kind            = $node.kind
        url             = $node.url
        text_file       = $node.text_file
        discovered_from = @($node.discovered_from | Sort-Object)
        fetch_status    = $node.fetch_status
        failure_count   = $node.failure_count
        last_http_status = $node.last_http_status
        last_error      = $node.last_error
        last_attempt_utc = $node.last_attempt_utc
        link_count      = (@($node.links | Where-Object { $titleSet.Contains($_) -and $_ -ne $title })).Count
    }
}

$finalCounts = Get-NodeFetchCounts -Nodes $nodes
$manifest = [pscustomobject]@{
    root_category       = $rootCategory
    generated_at_utc    = (Get-Date).ToUniversalTime().ToString('o')
    node_count          = $nodes.Count
    category_count      = (@($nodes.Values | Where-Object { $_.kind -eq 'category' })).Count
    page_count          = (@($nodes.Values | Where-Object { $_.kind -eq 'page' })).Count
    in_scope_edge_count = $edges.Count
    completed_count     = $finalCounts.completed
    failed_count        = $finalCounts.failed
    pending_count       = $finalCounts.pending
    output_directory    = (Resolve-Path $OutputDir).Path
}

$pageIndexPath = Join-Path $OutputDir 'page_index.json'
$linksPath = Join-Path $OutputDir 'links.csv'
$manifestPath = Join-Path $OutputDir 'manifest.json'
$failedPagesPath = Join-Path $OutputDir 'failed_pages.csv'

$pageIndex | ConvertTo-Json -Depth 6 | Set-Content -Path $pageIndexPath -Encoding UTF8
$edges | Export-Csv -Path $linksPath -NoTypeInformation -Encoding UTF8
$pageIndex | Where-Object { $_.fetch_status -eq 'failed' } | Export-Csv -Path $failedPagesPath -NoTypeInformation -Encoding UTF8
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host ''
Write-Host 'Scrape finished.'
Write-Host "  Manifest:   $manifestPath"
Write-Host "  Page index: $pageIndexPath"
Write-Host "  Links CSV:  $linksPath"
Write-Host "  Failures:   $failedPagesPath"
Write-Host "  Progress:   $progressDir"
Write-Host "  Text files: $textsDir"
