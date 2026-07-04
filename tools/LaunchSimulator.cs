using System;
using System.Diagnostics;
using System.IO;

internal static class LaunchSimulator
{
    [STAThread]
    private static int Main()
    {
        var root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar
        );
        var info = new ProcessStartInfo
        {
            FileName = @"C:\il\venv\Scripts\python.exe",
            Arguments = "\"" + Path.Combine(root, "run_sim.py") + "\" --max-fuel 456",
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        info.EnvironmentVariables["OMNI_KIT_ACCEPT_EULA"] = "YES";
        var process = Process.Start(info);
        if (process == null) return 1;
        var logPath = Path.Combine(root, "runs", "gui.log");
        Directory.CreateDirectory(Path.GetDirectoryName(logPath));
        var stdout = process.StandardOutput.ReadToEndAsync();
        var stderr = process.StandardError.ReadToEndAsync();
        process.WaitForExit();
        File.WriteAllText(logPath, stdout.Result + Environment.NewLine + stderr.Result);
        return process.ExitCode;
    }
}
