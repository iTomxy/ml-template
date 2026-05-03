' syncthing_bg.vbs
' Run Syncthing in the background without opening a console window or browser.
' NOTE:
' 1. use full path for robustness.
' 2. wrap with triple quotes in case of spaces in the path.
' 3. run `where syncthing` to find the path if you have it in your PATH environment variable.

strProcess = "syncthing.exe"
strCommand = """C:\Program Files\Syncthing\syncthing.exe"" --no-console --no-browser"

Set WshShell = CreateObject("WScript.Shell")
Set objWMIService = GetObject("winmgmts:\\.\root\cimv2")

' Query the system for any running process with the target name
Set colItems = objWMIService.ExecQuery("Select * from Win32_Process Where Name = '" & strProcess & "'")

' Only launch if the count is 0
If colItems.Count = 0 Then
    WshShell.Run strCommand, 0
End If

Set WshShell = Nothing
Set objWMIService = Nothing
