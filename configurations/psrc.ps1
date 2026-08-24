# PowerShell Resource Configuration file
# run `$profile` in powershell to see the path to this file

function prompt {
    "$($ExecutionContext.SessionState.Path.CurrentLocation.Path)`nPS> "
}

# let tab complete like bash: complete to common prefix
Set-PSReadLineKeyHandler -Key Tab -Function Complete
Set-Alias python3 python

# ask for confirmation before deleting: `rm`, `del`, `erase`, `rd`, `ri`, `rmdir`
# are all aliases of Remove-Item, so this covers every spelling
$PSDefaultParameterValues['Remove-Item:Confirm'] = $true
