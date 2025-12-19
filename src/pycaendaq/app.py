import os
import subprocess
import threading
import queue
import time
import signal
import re
import io
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, send_file
import matplotlib.pyplot as plt
import lgdo.lh5 as lh5
import numpy as np
from scipy.signal import periodogram


# When installed as a package, Flask needs to know where its templates and static files are.
# We'll set these dynamically in the run_app function.
app = Flask(__name__)

# --- Logging Setup ---
# LOG_DIR will now be relative to the *installed* package, or a user-defined location.
# For simplicity during development and packaging, we'll keep it relative to the app's root.
# In a production deployment, you might want this path to be configurable or point to /var/log.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
log_filepath = os.path.join(LOG_DIR, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath), # Log to file
        logging.StreamHandler() # Also log to console
    ]
)
# Get a logger instance
app_logger = logging.getLogger(__name__)
# --- End Logging Setup ---


# Global variables to manage the subprocess and its output
daq_process = None
output_queue = queue.Queue() # This queue will now be fed by messages passed through the logger
output_thread = None
process_lock = threading.Lock()
total_channels = 64

# Global to store the base name of the last acquired file
last_output_file_basename = None

# PLOT_DIR will be set dynamically in run_app to be relative to the installed package's static folder
PLOT_DIR = None # Initialize as None, will be set by run_app

def read_subprocess_output(process, output_q):
    """Reads stdout and stderr from the subprocess and puts them into a queue."""
    for line in process.stdout:
        # Log to file and console via the logger
        app_logger.info(line.rstrip('\r\n'))
        output_q.put(line.rstrip('\r\n')) # Still send to queue for SSE

    for line in process.stderr:
        error_message_processed = line.rstrip('\r\n')
        # Log errors
        app_logger.error("ERROR: " + error_message_processed)
        output_q.put("ERROR: " + error_message_processed)

    process.stdout.close()
    process.stderr.close()

    process.wait()

    exit_message = f"__PROCESS_EXITED__:{process.returncode}"
    app_logger.info(exit_message) # Log the exit message
    output_q.put(exit_message)
    app_logger.debug("Read thread finished and put __PROCESS_EXITED__:%s", process.returncode) # Use debug for internal server messages
    output_q.put(None)
    app_logger.debug("Read thread put None sentinel.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/start_acquisition', methods=['POST'])
def start_acquisition():
    global daq_process, output_thread, output_queue, last_output_file_basename

    with process_lock:
        if daq_process and daq_process.poll() is None:
            app_logger.warning('Acquisition already running.')
            return jsonify({'status': 'error', 'message': 'Acquisition already running.'}), 409

        data = request.json
        dig_address = data.get('dig_address')
        config_file = data.get('config_file')
        out_file = data.get('out_file')
        temperature = data.get('temperature')
        n_events = data.get('n_events')
        duration = data.get('duration')

        if not dig_address:
            app_logger.error('Digitizer Address cannot be empty.')
            return jsonify({'status': 'error', 'message': 'Digitizer Address cannot be empty.'}), 400
        if not config_file:
            app_logger.error('Configuration File cannot be empty.')
            return jsonify({'status': 'error', 'message': 'Configuration File cannot be empty.'}), 400
        
        # When installed, config_file might be relative to where the app is run,
        # or it might be a path to a config file *outside* the installed package.
        # For now, assume it's a path the user provides that needs to be checked.
        # If config files are also part of the package data, this logic needs adjustment.
        if not os.path.exists(config_file):
            app_logger.error(f"Configuration file not found: {config_file}")
            return jsonify({'status': 'error', 'message': f"Configuration file not found: {config_file}"}), 400
        
        if not out_file:
            app_logger.error('Output File Base Name cannot be empty.')
            return jsonify({'status': 'error', 'message': 'Output File Base Name cannot be empty.'}), 400

        # Ensure output file directory exists
        out_file_dir = os.path.dirname(out_file)
        if out_file_dir and not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir, exist_ok=True)
            app_logger.info(f"Created output directory: {out_file_dir}")


        # Store the base name of the output file for later plotting without manual path
        last_output_file_basename = os.path.join(out_file_dir, os.path.basename(out_file).replace(".lh5", ""))
        app_logger.info(f"Last output file basename set to: {last_output_file_basename}")


        # IMPORTANT: When running daq_scope.py as part of an installed package,
        # you should invoke it as a module using `python -m`.
        # This ensures Python finds it correctly within the installed package.
        command = ["python", "-m", "pycaendaq.daq-scope"] # Changed from "daq_scope.py"
        command.extend(["-a", dig_address])
        command.extend(["-c", config_file])
        command.extend(["-o", out_file])

        if temperature:
            command.append("-tt")
        if n_events:
            try:
                int(n_events)
                command.extend(["-n", str(n_events)]) # Ensure it's a string for subprocess
            except ValueError:
                app_logger.error('Number of Events must be an integer.')
                return jsonify({'status': 'error', 'message': 'Number of Events must be an integer.'}), 400
        if duration:
            try:
                int(duration)
                command.extend(["-d", str(duration)]) # Ensure it's a string for subprocess
            except ValueError:
                app_logger.error('Duration must be an integer.')
                return jsonify({'status': 'error', 'message': 'Duration must be an integer.'}), 400

        try:
            # Clear queue before starting new process
            while not output_queue.empty():
                output_queue.get_nowait()

            daq_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            output_thread = threading.Thread(target=read_subprocess_output, args=(daq_process, output_queue))
            output_thread.daemon = True
            output_thread.start()

            log_start_message = f"Starting acquisition with command: {' '.join(command)}"
            app_logger.info(log_start_message)
            output_queue.put(log_start_message) # Also push to queue for SSE

            return jsonify({'status': 'success', 'message': 'Acquisition started.'})
        except FileNotFoundError:
            # This error might occur if 'python' is not in PATH or if the module path is wrong.
            app_logger.exception('Python executable or pycaendaq.daq-scope module not found.')
            return jsonify({'status': 'error', 'message': 'Python executable or pycaendaq.daq-scope module not found.'}), 500
        except Exception as e:
            app_logger.exception(f"Failed to start acquisition: {e}")
            return jsonify({'status': 'error', 'message': f"Failed to start acquisition: {e}"}), 500

@app.route('/stop_acquisition', methods=['POST'])
def stop_acquisition():
    global daq_process, output_thread

    with process_lock:
        if daq_process and daq_process.poll() is None:
            stop_message = "Stopping acquisition..."
            app_logger.info(stop_message)
            output_queue.put(stop_message)
            try:
                daq_process.send_signal(signal.SIGINT)
                daq_process.wait(timeout=5)
                app_logger.info("Acquisition process gracefully terminated.")
            except subprocess.TimeoutExpired:
                app_logger.warning("Process did not terminate gracefully, forcing kill.")
                daq_process.kill()
            except Exception as e:
                app_logger.exception(f"Error while trying to stop process: {e}")
            finally:
                return jsonify({'status': 'success', 'message': 'Attempted to stop acquisition.'})
        else:
            app_logger.info('No acquisition is currently running.')
            return jsonify({'status': 'info', 'message': 'No acquisition is currently running.'})

@app.route('/stream_log')
def stream_log():
    """Streams live log updates to the browser using Server-Sent Events."""
    def generate():
        while True:
            try:
                message = output_queue.get(timeout=1)

                if message is None:
                    app_logger.debug("Generator received None sentinel. Closing stream.")
                    time.sleep(0.1) # Give a tiny bit of time for client to receive last message
                    break

                # The client-side JavaScript expects a specific format for exit messages.
                # The timestamping happens client-side for these real-time messages.
                if message.startswith("__PROCESS_EXITED__"):
                    app_logger.debug("Generator got %s", message)
                    # The client side uses this for final status. Do not add server-side timestamp here.
                    yield f"data: {message}\n\n"
                else:
                    yield f"data: {message}\n\n"
            except queue.Empty:
                yield "data: \n\n" # Keep connection alive
            except Exception as e:
                app_logger.exception(f"Error in stream_log generator: {e}")
                break

    return Response(generate(), mimetype='text/event-stream')

def find_latest_lh5_file(base_path):
    if not base_path:
        return None

    file_dir = os.path.dirname(base_path)
    file_basename_only = os.path.basename(base_path)

    # Regex to match the timestamp format: _YYYYMMDDTHHMMSSZ.lh5
    pattern = re.compile(rf"^{re.escape(file_basename_only)}_\d{{8}}T\d{{6}}Z\.lh5$")

    latest_file = None
    latest_timestamp = -1

    if not os.path.isdir(file_dir):
        app_logger.warning(f"Directory not found for base_path: {file_dir}")
        return None

    for filename in os.listdir(file_dir):
        if pattern.match(filename):
            full_path = os.path.join(file_dir, filename)
            try:
                mod_time = os.path.getmtime(full_path)
                if mod_time > latest_timestamp:
                    latest_timestamp = mod_time
                    latest_file = full_path
            except Exception as e:
                app_logger.warning(f"Could not get modification time for {full_path}: {e}")
                continue
    return latest_file

@app.route('/plot_waveforms', methods=['POST'])
def plot_waveforms():
    data = request.json
    lh5_file_param = data.get('lh5_file')
    plot_last = data.get('plot_last', False)
    plot_fft = data.get('plot_fft', False)

    resolved_lh5_file = None

    if not lh5_file_param:
        global last_output_file_basename
        if last_output_file_basename:
            resolved_lh5_file = find_latest_lh5_file(last_output_file_basename)
            if not resolved_lh5_file:
                app_logger.error(f"No recent LH5 file found matching '{last_output_file_basename}_YYYYMMDDTHHMMSSZ.lh5'.")
                return jsonify({'status': 'error', 'message': f"No recent LH5 file found matching '{last_output_file_basename}_YYYYMMSSZ.lh5'."}), 404
            app_logger.info(f"Auto-discovered latest LH5 file: {resolved_lh5_file}")
        else:
            app_logger.error('No LH5 file path provided and no previous acquisition found to infer from.')
            return jsonify({'status': 'error', 'message': 'No LH5 file path provided and no previous acquisition found to infer from.'}), 400
    else:
        resolved_lh5_file = lh5_file_param

    # When the app is installed, app.root_path might point to the site-packages directory.
    # We need to ensure paths are handled correctly.
    # For now, assuming resolved_lh5_file is an absolute path or relative to the current working directory.
    # If it's relative to the app's installed location, further path adjustments might be needed.
    if not os.path.isabs(resolved_lh5_file):
        # This assumes the user provides relative paths from the current working directory
        # where the 'daq-web' command is executed.
        abs_lh5_file = os.path.abspath(resolved_lh5_file)
    else:
        abs_lh5_file = resolved_lh5_file

    if not os.path.exists(abs_lh5_file):
        app_logger.error(f"LH5 file not found: {resolved_lh5_file}")
        return jsonify({'status': 'error', 'message': f"LH5 file not found: {resolved_lh5_file}"}), 404

    try:
        channels = lh5.ls(abs_lh5_file)

        plot_channels = []
        for i in range(total_channels):
            chn = f"ch{i:03d}"
            if chn in channels:
                plot_channels.append(chn)

        if not plot_channels:
            app_logger.warning('No plotable channels found in the LH5 file.')
            return jsonify({'status': 'error', 'message': 'No plotable channels found in the LH5 file.'}), 400

        ncols = 4
        nrows = len(plot_channels) // ncols + (len(plot_channels) % ncols > 0)

        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))
        axes_flat = axes.flatten()

        for idx, chn in enumerate(plot_channels):
            ax = axes_flat[idx]
            try:
                if plot_fft:
                    raw_data = lh5.read(f"{chn}/raw", abs_lh5_file, n_rows=500)
                    if raw_data is None or not hasattr(raw_data, 'waveform') or raw_data.waveform is None:
                        ax.set_title(f"{chn} (No WFs)")
                        continue
                    wfs = raw_data.waveform.values.nda
                    nev, wsize = wfs.shape
                    dt = raw_data.waveform.dt.nda[0]
                    rate =  1 / dt * 1e9 # rate in Hz
                    #dts = np.linspace(0, (wsize-1) * dt, wsize) # Not used for FFT plot directly
                    psd = np.zeros(wsize // 2 + 1) # Initialize PSD accumulator
                    for j in range(min(100, nev)):
                        freq_tmp, psd_tmp = periodogram(wfs[j], rate, scaling='density')
                        psd += psd_tmp
                    psd = np.array(psd / min(100, nev)) # Average PSD
                    freq = np.array(freq_tmp)
                    rms = np.sqrt(np.trapz(psd,freq))
                    ax.plot(freq[1:], psd[1:])
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlim(freq[1],freq[-1])
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel(r"Power Spectral Density ([ADC$^2$/Hz])")
                    ax.set_title(f"{chn} (RMS = {rms:.1f} LSB)")

                elif plot_last:
                    n_total_rows = lh5.read_n_rows(f"{chn}/raw", abs_lh5_file)
                    if n_total_rows == 0:
                        ax.set_title(f"{chn} (No WFs)")
                        continue
                    row_offset = max(0, n_total_rows - 1)
                    raw_data = lh5.read(f"{chn}/raw", abs_lh5_file, start_row=row_offset, n_rows=1)
                    if raw_data is None or not hasattr(raw_data, 'waveform') or raw_data.waveform is None:
                        ax.set_title(f"{chn} (No WFs)")
                        continue
                    wfs = raw_data.waveform.values.nda
                    nev, wsize = wfs.shape
                    dt = raw_data.waveform.dt.nda[0] / 1000. # us
                    timestamp = raw_data.timestamp.nda
                    dts = np.linspace(0, (wsize-1) * dt, wsize)
                    if nev == 1 and wsize > 0:
                        ax.plot(dts, wfs[0])
                        ax.set_xlim(0, dts[-1])
                        ax.set_xlabel(r"Time ($\mu$s)")
                        ax.set_ylabel("ADC")
                        date = datetime.fromtimestamp(timestamp[0]/1e9).strftime("%Y-%m-%d %H:%M:%S")
                        ax.set_title(f"{chn} (Last Event at {date})")
                    else:
                        ax.set_title(f"{chn} (No Last Event)")
                else: # Plot first 10 events (default)
                    raw_data = lh5.read(f"{chn}/raw", abs_lh5_file, n_rows=10)
                    if raw_data is None or not hasattr(raw_data, 'waveform') or raw_data.waveform is None:
                        ax.set_title(f"{chn} (No WFs)")
                        continue
                    wfs = raw_data.waveform.values.nda
                    nev, wsize = wfs.shape
                    dt = raw_data.waveform.dt.nda[0] / 1000. # us
                    dts = np.linspace(0, (wsize-1) * dt, wsize)
                    for j in range(min(10, nev)):
                        ax.plot(dts, wfs[j])
                        ax.set_xlim(0, dts[-1])
                        ax.set_xlabel(r"Time ($\mu$s)")
                        ax.set_ylabel("ADC")
                    ax.set_title(f"{chn} (First {min(10, nev)} Events)")
            except Exception as plot_e:
                app_logger.exception(f"Error plotting channel {chn}")
                ax.set_title(f"{chn} (Error)")
                ax.text(0.5, 0.5, "Plot Error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue

        for i in range(len(plot_channels), len(axes_flat)):
            fig.delaxes(axes_flat[i])


        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plt.close(fig)

        response = Response(buffer.getvalue(), mimetype='image/png')
        response.headers['X-LH5-File-Path'] = resolved_lh5_file
        app_logger.info(f"Waveform plot generated for {resolved_lh5_file}")
        return response

    except Exception as e:
        app_logger.exception(f"Error plotting waveforms: {e}")
        return jsonify({'status': 'error', 'message': f"Error plotting waveforms: {str(e)}"}), 500

def run_app():
    """
    Entry point function to run the Flask web application.
    This function should be called when the package is installed and run
    via the 'daq-web' console script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app.template_folder = os.path.join(current_dir, 'templates')
    app.static_folder = os.path.join(current_dir, 'static')

    # Ensure the PLOT_DIR is also correctly set relative to the static folder
    global PLOT_DIR
    PLOT_DIR = os.path.join(app.static_folder, 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)
    app_logger.info(f"Flask app template_folder: {app.template_folder}")
    app_logger.info(f"Flask app static_folder: {app.static_folder}")
    app_logger.info(f"Flask app PLOT_DIR: {PLOT_DIR}")

    app_logger.info("Starting Flask web application...")
    app.run(debug=True, port=44500)

if __name__ == '__main__':
    run_app()

