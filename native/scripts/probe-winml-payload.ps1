# 摸清 WinML 2.x 包里到底带了什么（EP、ORT 版本）
$ErrorActionPreference = 'SilentlyContinue'
$out = Join-Path $PSScriptRoot '..\third_party\winml_payload.txt'
$L = New-Object System.Collections.Generic.List[string]

$roots = @(
    'C:\Program Files\WindowsApps\Microsoft.WindowsAppRuntime.2_2.4.0.0_x64__8wekyb3d8bbwe',
    'C:\Windows\SystemApps\Microsoft.WindowsAppRuntime.CBS.2_8wekyb3d8bbwe'
)

foreach ($r in $roots) {
    $L.Add("=== $r ===")
    if (-not (Test-Path $r)) { $L.Add('  (not present)'); continue }
    Get-ChildItem -Path $r -Recurse -File -Include *.dll, *.json, *.xml -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        ForEach-Object {
            $rel = $_.FullName.Substring($r.Length)
            $v = $_.VersionInfo.FileVersion
            if ($v) { $L.Add(("  {0,-58} {1}" -f $rel, $v)) } else { $L.Add(("  {0,-58} {1}" -f $rel, ('{0} KB' -f [math]::Round($_.Length/1KB,0)))) }
        }
    $L.Add('')
}

$L.Add('=== 已安装的 WinML / ONNX EP 包 ===')
$ep = Get-AppxPackage | Where-Object { $_.Name -match 'WinML|ONNX|TensorRt|DirectML' } | Sort-Object Name, Version -Unique
if ($ep) { $ep | ForEach-Object { $L.Add("$($_.Name)  $($_.Version)") } } else { $L.Add('(none)') }

$L.Add('')
$L.Add('=== EP catalog 相关的 Windows API 是否可查 ===')
foreach ($p in @('C:\Windows\System32\Windows.AI.MachineLearning.dll',
                 'C:\Windows\System32\WinML.dll')) {
    if (Test-Path $p) { $L.Add("FOUND $p") } else { $L.Add("MISSING $p") }
}

$L | Set-Content -Path $out -Encoding UTF8
Write-Output "written: $out"
