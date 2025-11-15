Set oShell = CreateObject("WScript.Shell")
strDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
oShell.Run Chr(34) & strDir & "\start_demo.bat" & Chr(34), 1, False
