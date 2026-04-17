# PowerShell Resource Configuration file
# run `$profile` in powershell to see the path to this file

function prompt {
    "$($ExecutionContext.SessionState.Path.CurrentLocation.Path)`nPS> "
}
