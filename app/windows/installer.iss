; 玄石 GoZero Windows 安裝程式（Inno Setup 6，GitHub windows runner 內建）
; CI 用法：ISCC.exe /DAppVersion=1.0.0 installer.iss
; 檔名/顯示名維持 ASCII，避免 ISCC 對非 UTF-8-BOM 檔案的編碼歧義。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{8E1F6A02-5C7D-4B7E-9A34-2F60C41D7B19}
AppName=XuanShi GoZero
AppVersion={#AppVersion}
DefaultDirName={autopf}\XuanShi GoZero
DefaultGroupName=XuanShi GoZero
; 裝到使用者層級即可，不要求系統管理員
PrivilegesRequired=lowest
OutputDir=.
OutputBaseFilename=gozero_go-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\gozero_go.exe

[Files]
; Flutter Release 整包：exe + flutter_windows.dll + data\，缺一不可
Source: "..\build\windows\x64\runner\Release\*"; DestDir: "{app}"; \
  Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\XuanShi GoZero"; Filename: "{app}\gozero_go.exe"
Name: "{autodesktop}\XuanShi GoZero"; Filename: "{app}\gozero_go.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "建立桌面捷徑"; GroupDescription: "其他："

[Run]
Filename: "{app}\gozero_go.exe"; Description: "安裝完成後啟動"; \
  Flags: nowait postinstall skipifsilent
