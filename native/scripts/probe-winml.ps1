# Windows ML 可用性探测（只读，不改任何东西）
# 输出写文件，避免中文终端转码问题
$ErrorActionPreference = 'SilentlyContinue'
$out = Join-Path $PSScriptRoot '..\third_party\winml_probe.txt'
$L = New-Object System.Collections.Generic.List[string]

$L.Add('=== OS ===')
$os = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$L.Add("Build=$($os.CurrentBuild).$($os.UBR)  DisplayVersion=$($os.DisplayVersion)  ProductName=$($os.ProductName)")

$L.Add('')
$L.Add('=== WindowsAppRuntime / ML packages ===')
$pkgs = Get-AppxPackage | Where-Object { $_.Name -match 'WindowsAppRuntime|WinML|MachineLearning' } |
        Sort-Object Name, Version -Unique
foreach ($p in $pkgs) {
    $L.Add("$($p.Name)  $($p.Version)")
    $L.Add("    loc: $($p.InstallLocation)")
}

$L.Add('')
$L.Add('=== Microsoft.Windows.AI.MachineLearning.dll lookup ===')
$candidates = @(
    'C:\Windows\System32\Microsoft.Windows.AI.MachineLearning.dll',
    'C:\Windows\SysWOW64\Microsoft.Windows.AI.MachineLearning.dll'
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $L.Add("FOUND $c  ver=$((Get-Item $c).VersionInfo.FileVersion)") }
    else { $L.Add("MISSING $c") }
}

# Windows ML 2.x 的运行时通常落在 WindowsAppRuntime 2.x 包的目录里
foreach ($p in $pkgs) {
    if (-not $p.InstallLocation) { continue }
    $hit = Get-ChildItem -Path $p.InstallLocation -Filter 'Microsoft.Windows.AI.MachineLearning*.dll' -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 3
    foreach ($h in $hit) { $L.Add("FOUND $($h.FullName)  ver=$($h.VersionInfo.FileVersion)") }
}

$L.Add('')
$L.Add('=== onnxruntime.dll on system (WinML hosts one) ===')
foreach ($p in $pkgs) {
    if (-not $p.InstallLocation) { continue }
    $hit = Get-ChildItem -Path $p.InstallLocation -Filter 'onnxruntime*.dll' -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 5
    foreach ($h in $hit) { $L.Add("FOUND $($h.FullName)") }
}

$L | Set-Content -Path $out -Encoding UTF8
Write-Output "written: $out"
