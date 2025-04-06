# Launch a message box with PowerShell similarly to the msg.exe in Windows 10 Pro.
## Usage
# powershell -ExecutionPolicy Bypass -File msg.ps1 -Message "<SOME_MESSAGE>"

param (
    [string]$Message = ""
)
# Exit if the message is empty or just whitespace
if ([string]::IsNullOrWhiteSpace($Message)) {
    return
}

# Get current time
$time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Create formatted message
$fullMessage = "[$time] $Message"

# Show message box
Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show($fullMessage, $time, 'OK', 'Information')
