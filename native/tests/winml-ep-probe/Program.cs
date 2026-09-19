using Microsoft.Windows.AI.MachineLearning;

// usage:
//   (no args)                -> list catalog
//   --install [EPNAME]       -> download + install + register an EP
//                               (default: NvTensorRTRTXExecutionProvider)

var cliArgs = Environment.GetCommandLineArgs().Skip(1).ToArray();

Console.WriteLine("=== WinML ExecutionProviderCatalog probe ===");
Console.WriteLine($"framework : {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}");
Console.WriteLine($"OS build  : {Environment.OSVersion.Version}");
Console.WriteLine($"arch      : {System.Runtime.InteropServices.RuntimeInformation.OSArchitecture}");
Console.WriteLine();

var catalog = ExecutionProviderCatalog.GetDefault();
var providers = catalog.FindAllProviders();

Console.WriteLine($"catalog ({providers.Length} entries):");
foreach (var p in providers)
{
    Console.WriteLine($"  name={p.Name,-32} readyState={p.ReadyState}");
}
Console.WriteLine();

if (cliArgs.Length == 0 || cliArgs[0] != "--install")
{
    Console.WriteLine("(pass --install <EPNAME> to download / install / register a provider)");
    return 0;
}

var wanted = cliArgs.Length > 1 ? cliArgs[1] : "NvTensorRTRTXExecutionProvider";
var target = providers.FirstOrDefault(p => p.Name == wanted);
if (target is null)
{
    Console.WriteLine($"!! provider '{wanted}' is not offered by the catalog on this machine");
    return 2;
}

Console.WriteLine($"--- installing '{target.Name}' (readyState={target.ReadyState}) ---");
Console.WriteLine("downloads from Windows Update; may take several minutes");
Console.WriteLine();

try
{
    var op = target.EnsureReadyAsync();
    op.Progress = (_, progress) => Console.WriteLine($"    progress: {progress:0.#}%");

    var result = await op;

    Console.WriteLine();
    Console.WriteLine($"EnsureReadyAsync -> Status={result.Status}");
    Console.WriteLine($"  ExtendedError  = {result.ExtendedError}");
    Console.WriteLine($"  DiagnosticText = {result.DiagnosticText}");

    if (result.Status != ExecutionProviderReadyResultState.Success)
    {
        Console.WriteLine("!! not successful -> aborting before TryRegister");
        return 3;
    }

    Console.WriteLine();
    Console.WriteLine("--- TryRegister() ---");
    bool registered = target.TryRegister();
    Console.WriteLine($"TryRegister -> {registered}");

    try
    {
        Console.WriteLine($"  LibraryPath = {target.LibraryPath}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  (LibraryPath unavailable: {ex.GetType().Name})");
    }

    Console.WriteLine();
    Console.WriteLine($"after install, readyState = {target.ReadyState}");
}
catch (Exception ex)
{
    Console.WriteLine("INSTALL FAILED: " + ex.GetType().FullName);
    Console.WriteLine("  message: " + ex.Message);
    if (ex.InnerException is not null)
        Console.WriteLine("  inner  : " + ex.InnerException.GetType().FullName + ": " + ex.InnerException.Message);
    return 4;
}

Console.WriteLine();
Console.WriteLine("=== done ===");
return 0;

