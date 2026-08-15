# PowerShell Resource Configuration file
# run `$profile` in powershell to see the path to this file

function prompt {
    "$($ExecutionContext.SessionState.Path.CurrentLocation.Path)`nPS> "
}

# let tab complete like bash: complete to common prefix
Set-PSReadLineKeyHandler -Key Tab -Function Complete
Set-Alias python3 python
