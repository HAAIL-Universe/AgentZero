Set WShell = CreateObject("WScript.Shell")
WShell.Run "powershell.exe -Command ""Start-Process cmd -ArgumentList '/k python Z:\AgentZero\launch.py' -WindowStyle Normal""", 0, False
