import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import os
import signal # Import the signal module for SIGINT

class DAQController:
    def __init__(self, master):
        self.master = master
        master.title("DAQ Scope Controller")

        self.process = None  # To hold the subprocess object
        self.output_thread = None # To hold the thread for reading stdout

        # --- Input Widgets ---
        row = 0

        tk.Label(master, text="Digitizer Address:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.dig_address_entry = tk.Entry(master, width=50)
        self.dig_address_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        self.dig_address_entry.insert(0, "dig2://caendgtz-usb-52696") # Default value
        row += 1

        tk.Label(master, text="Config File:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.config_file_entry = tk.Entry(master, width=40)
        self.config_file_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.config_file_button = tk.Button(master, text="Browse", command=self.browse_config_file)
        self.config_file_button.grid(row=row, column=2, padx=5, pady=2)
        row += 1

        tk.Label(master, text="Output File (Base Name):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.out_file_entry = tk.Entry(master, width=50)
        self.out_file_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        self.out_file_entry.insert(0, "data_acquisition") # Default value
        row += 1

        self.temperature_var = tk.BooleanVar()
        self.temperature_check = tk.Checkbutton(master, text="Save Temperature", var=self.temperature_var)
        self.temperature_check.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        row += 1

        tk.Label(master, text="Number of Events (optional):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.n_events_entry = tk.Entry(master, width=10)
        self.n_events_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        tk.Label(master, text="Duration (seconds, optional):").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.duration_entry = tk.Entry(master, width=10)
        self.duration_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        # --- Control Buttons ---
        self.start_button = tk.Button(master, text="Start Acquisition", command=self.start_acquisition, bg="green", fg="white")
        self.start_button.grid(row=row, column=0, padx=5, pady=10, sticky="ew")

        self.stop_button = tk.Button(master, text="Stop Acquisition", command=self.stop_acquisition, bg="red", fg="white", state=tk.DISABLED)
        self.stop_button.grid(row=row, column=1, padx=5, pady=10, sticky="ew")
        row += 1

        # --- Output Log ---
        tk.Label(master, text="Acquisition Log:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        row += 1
        self.log_text = tk.Text(master, height=15, width=80, state=tk.DISABLED, bg="black", fg="lime green")
        self.log_text.grid(row=row, column=0, columnspan=3, padx=5, pady=2, sticky="nsew")
        self.log_text_scrollbar = tk.Scrollbar(master, command=self.log_text.yview)
        self.log_text_scrollbar.grid(row=row, column=3, sticky="ns")
        self.log_text['yscrollcommand'] = self.log_text_scrollbar.set

        master.grid_rowconfigure(row, weight=1)
        master.grid_columnconfigure(1, weight=1)

    def browse_config_file(self):
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=(("YAML files", "*.yaml"), ("All files", "*.*"))
        )
        if filename:
            self.config_file_entry.delete(0, tk.END)
            self.config_file_entry.insert(0, filename)

    def update_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END) # Auto-scroll to the end
        self.log_text.config(state=tk.DISABLED)

    def read_output(self):
        # Read stdout
        for line in self.process.stdout:
            self.master.after(0, self.update_log, line) # No .decode() needed with text=True

        # Read stderr
        for line in self.process.stderr:
            self.master.after(0, self.update_log, f"ERROR: {line}") # No .decode() needed with text=True

        self.process.wait() # Wait for the process to terminate
        self.master.after(0, self.on_acquisition_end)

    def on_acquisition_end(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.process and self.process.returncode == 0:
            self.update_log("\nAcquisition finished successfully.\n")
        elif self.process and self.process.returncode is not None:
            self.update_log(f"\nAcquisition stopped with exit code: {self.process.returncode}\n")
        else:
            self.update_log("\nAcquisition process terminated unexpectedly.\n")
        self.process = None # Clear the process reference

    def start_acquisition(self):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Warning", "Acquisition is already running.")
            return

        dig_address = self.dig_address_entry.get().strip()
        config_file = self.config_file_entry.get().strip()
        out_file = self.out_file_entry.get().strip()
        temperature = self.temperature_var.get()
        n_events = self.n_events_entry.get().strip()
        duration = self.duration_entry.get().strip()

        if not dig_address:
            messagebox.showerror("Error", "Digitizer Address cannot be empty.")
            return
        if not config_file:
            messagebox.showerror("Error", "Configuration File cannot be empty.")
            return
        if not os.path.exists(config_file):
            messagebox.showerror("Error", f"Configuration file not found: {config_file}")
            return
        if not out_file:
            messagebox.showerror("Error", "Output File Base Name cannot be empty.")
            return

        # MODIFIED LINE: Added '-u' flag to the Python command
        command = ["python", "-u", "daq_scope.py"]
        command.extend(["-a", dig_address])
        command.extend(["-c", config_file])
        command.extend(["-o", out_file])

        if temperature:
            command.append("-tt")
        if n_events:
            try:
                int(n_events)
                command.extend(["-n", n_events])
            except ValueError:
                messagebox.showerror("Error", "Number of Events must be an integer.")
                return
        if duration:
            try:
                int(duration)
                command.extend(["-d", duration])
            except ValueError:
                messagebox.showerror("Error", "Duration must be an integer.")
                return

        self.update_log(f"Starting acquisition with command: {' '.join(command)}\n")
        
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1, # This will now work correctly with text=True and -u flag
                universal_newlines=True
            )
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            self.output_thread = threading.Thread(target=self.read_output)
            self.output_thread.daemon = True
            self.output_thread.start()

        except FileNotFoundError:
            messagebox.showerror("Error", "Python executable or daq_scope.py not found. Ensure they are in your PATH or current directory.")
            self.on_acquisition_end()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start acquisition: {e}")
            self.on_acquisition_end()

    def stop_acquisition(self):
        if self.process and self.process.poll() is None: # Check if the process is still running
            self.update_log("\nStopping acquisition...\n")
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.update_log("Process did not terminate gracefully, forcing kill.\n")
                self.process.kill()
            except Exception as e:
                self.update_log(f"Error while trying to stop process: {e}\n")
            finally:
                self.on_acquisition_end()
        else:
            messagebox.showinfo("Info", "No acquisition is currently running.")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = DAQController(root)
    root.mainloop()

if __name__ == "__main__":
    main()