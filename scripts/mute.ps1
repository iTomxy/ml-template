# mute.ps1
# To mute system volume
# Run:
#     powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File mute.ps1

# Create a Windows Shell object
$wshell = New-Object -ComObject wscript.shell

# Press the "Volume Down" key (character code 174) 50 times rapidly
# This guarantees the volume drops to exactly 0% no matter what it was set to
1..50 | ForEach-Object { $wshell.SendKeys([char]174) }
