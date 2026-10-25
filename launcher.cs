using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace SensumPATVIS
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string scriptPath = Path.Combine(baseDir, "run_patvis.py");

                if (!File.Exists(scriptPath))
                {
                    MessageBox.Show(
                        "Could not locate 'run_patvis.py'. Please ensure it exists in:\n" + baseDir,
                        "Sensum PATVIS APA",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return;
                }

                // Check possible Python interpreters
                string[] pythonCandidates = new string[]
                {
                    @"D:\Programs\miniforge\envs\deeplearning\pythonw.exe",
                    @"D:\Programs\miniforge\envs\deeplearning\python.exe",
                    Path.Combine(baseDir, "python", "pythonw.exe"),
                    Path.Combine(baseDir, "venv", "Scripts", "pythonw.exe"),
                    "pythonw.exe",
                    "python.exe"
                };

                string pythonExe = null;
                foreach (string candidate in pythonCandidates)
                {
                    if (Path.IsPathRooted(candidate))
                    {
                        if (File.Exists(candidate))
                        {
                            pythonExe = candidate;
                            break;
                        }
                    }
                    else
                    {
                        // Relative or in PATH
                        pythonExe = candidate;
                        break;
                    }
                }

                if (string.IsNullOrEmpty(pythonExe))
                {
                    pythonExe = @"D:\Programs\miniforge\envs\deeplearning\python.exe";
                }

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = pythonExe;
                psi.Arguments = "\"" + scriptPath + "\"";
                psi.WorkingDirectory = baseDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;

                // Ensure PATH and PYTHONPATH are clean
                string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                string pythonDir = Path.GetDirectoryName(pythonExe);
                if (!string.IsNullOrEmpty(pythonDir))
                {
                    string scriptsDir = Path.Combine(pythonDir, "Scripts");
                    string libraryBin = Path.Combine(pythonDir, "Library", "bin");
                    psi.EnvironmentVariables["PATH"] = baseDir + ";" + pythonDir + ";" + scriptsDir + ";" + libraryBin + ";" + existingPath;
                }
                psi.EnvironmentVariables["PYTHONPATH"] = baseDir;

                Process proc = Process.Start(psi);
                if (proc == null)
                {
                    MessageBox.Show("Failed to start process.", "Sensum PATVIS APA", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error launching Sensum PATVIS APA:\n" + ex.Message,
                    "Sensum PATVIS APA",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
