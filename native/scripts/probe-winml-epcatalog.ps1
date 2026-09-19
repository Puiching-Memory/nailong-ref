# 探 WinML EP catalog：workloads 清单 + ML dll 里的 EP 相关符号
$ErrorActionPreference = 'SilentlyContinue'
$out = Join-Path $PSScriptRoot '..\third_party\winml_ep_catalog.txt'
$L = New-Object System.Collections.Generic.List[string]

$pkg = 'C:\Program Files\WindowsApps\Microsoft.WindowsAppRuntime.2_2.4.0.0_x64__8wekyb3d8bbwe'

$L.Add('=== workloads*.json 里出现的 EP / 厂商关键字 ===')
Get-ChildItem -Path $pkg -Filter 'workloads*.json' | ForEach-Object {
    $L.Add("--- $($_.Name)  ($([math]::Round($_.Length/1KB,0)) KB) ---")
    $txt = Get-Content $_.FullName -Raw
    $hits = [regex]::Matches($txt, '[\w\.]*(?:ExecutionProvider|TensorRt|TensorRT|NvTensor|DirectML|QNN|OpenVINO|MIGraphX|VitisAI|WebGpu|EPVersion|ProviderName)[\w\.]*') |
            ForEach-Object { $_.Value } | Sort-Object -Unique
    if ($hits) { $hits | ForEach-Object { $L.Add("    $_") } } else { $L.Add('    (no EP keywords)') }
}

$L.Add('')
$L.Add('=== Microsoft.Windows.AI.MachineLearning.dll 内的 EP 相关字符串 ===')
$dll = Join-Path $pkg 'Microsoft.Windows.AI.MachineLearning.dll'
$L.Add("size: $([math]::Round((Get-Item $dll).Length/1MB,2)) MB")
$bytes = [IO.File]::ReadAllBytes($dll)
$text = [Text.Encoding]::Latin1.GetString($bytes)

$patterns = [ordered]@{
    'ExecutionProviderCatalog' = 'ExecutionProviderCatalog'
    'EnsureReadyAsync'         = 'EnsureReadyAsync'
    'EnsureAndRegisterCertifiedAsync' = 'EnsureAndRegisterCertifiedAsync'
    'GetEpDevices'             = 'GetEpDevices'
    'RegisterExecutionProviderLibrary' = 'RegisterExecutionProviderLibrary'
    'AppendExecutionProvider_V2' = 'AppendExecutionProvider_V2'
    'NvTensorRtRtx'            = 'NvTensorRtRtx'
    'TensorRTRtx'              = 'TensorRTRtx'
    'DirectML'                 = 'DirectML'
    'OpenVINO'                 = 'OpenVINO'
    'VitisAI'                  = 'VitisAI'
    'MIGraphX'                 = 'MIGraphX'
    'QNNExecutionProvider'     = 'QNNExecutionProvider'
    'onnxruntime_providers'    = 'onnxruntime_providers'
}
foreach ($k in $patterns.Keys) {
    $n = ([regex]::Matches($text, [regex]::Escape($patterns[$k]))).Count
    $L.Add(("  {0,-34} {1}" -f $k, $(if ($n -gt 0) { "HIT x$n" } else { '-' })))
}

$L | Set-Content -Path $out -Encoding UTF8
Write-Output "written: $out"
