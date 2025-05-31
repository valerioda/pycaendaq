import os
import subprocess
import threading
import queue
import time
import signal
import re
import io
import logging # Import logging module
from datetime import datetime # Import datetime for date-based log files

from flask import Flask, render_template, request, jsonify, Response, send_file
import matplotlib.pyplot as plt
import lgdo.lh5 as lh5
import numpy as np


app = Flask(__name__)

# --- Logging Setup ---
LOG_DIR = os.path.join(app.root_path, 'logs')
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

# Global to store the base name of the last acquired file
last_output_file_basename = None

PLOT_DIR = os.path.join(app.root_path, 'static', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

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
        config_file_path = os.path.join(app.root_path, config_file)
        if not os.path.exists(config_file_path):
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
                app_logger.error('Number of Events must be an integer.')
                return jsonify({'status': 'error', 'message': 'Number of Events must be an integer.'}), 400
        if duration:
            try:
                int(duration)
                command.extend(["-d", duration])
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
            app_logger.exception('Python executable or daq_scope.py not found.')
            return jsonify({'status': 'error', 'message': 'Python executable or daq_scope.py not found.'}), 500
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

    resolved_lh5_file = None

    if not lh5_file_param:
        global last_output_file_basename
        if last_output_file_basename:
            resolved_lh5_file = find_latest_lh5_file(last_output_file_basename)
            if not resolved_lh5_file:
                app_logger.error(f"No recent LH5 file found matching '{last_output_file_basename}_YYYYMMDDTHHMMSSZ.lh5'.")
                return jsonify({'status': 'error', 'message': f"No recent LH5 file found matching '{last_output_file_basename}_YYYYMMDDTHHMMSSZ.lh5'."}), 404
            app_logger.info(f"Auto-discovered latest LH5 file: {resolved_lh5_file}")
        else:
            app_logger.error('No LH5 file path provided and no previous acquisition found to infer from.')
            return jsonify({'status': 'error', 'message': 'No LH5 file path provided and no previous acquisition found to infer from.'}), 400
    else:
        resolved_lh5_file = lh5_file_param

    if not os.path.isabs(resolved_lh5_file):
        abs_lh5_file = os.path.abspath(os.path.join(app.root_path, resolved_lh5_file))
    else:
        abs_lh5_file = resolved_lh5_file

    if not os.path.exists(abs_lh5_file):
        app_logger.error(f"LH5 file not found: {resolved_lh5_file}")
        return jsonify({'status': 'error', 'message': f"LH5 file not found: {resolved_lh5_file}"}), 404

    try:
        channels = lh5.ls(abs_lh5_file)

        plot_channels = []
        for i in range(64):
            chn = f"ch{i:03d}"
            if chn in channels:
                plot_channels.append(chn)
            #if len(plot_channels) >= 6:
            #    break

        if not plot_channels:
            app_logger.warning('No plotable channels found in the LH5 file.')
            return jsonify({'status': 'error', 'message': 'No plotable channels found in the LH5 file.'}), 400

        fig, axes = plt.subplots(len(plot_channels) // 2 + (len(plot_channels) % 2 > 0), 2, figsize=(16, 6 * (len(plot_channels) // 2 + (len(plot_channels) % 2 > 0))))
        axes_flat = axes.flatten()

        for idx, chn in enumerate(plot_channels):
            ax = axes_flat[idx]
            try:
                raw_data = lh5.read(f"{chn}/raw", abs_lh5_file, n_rows=10)

                if raw_data is None or not hasattr(raw_data, 'waveform') or raw_data.waveform is None:
                    ax.set_title(f"{chn} (No WFs)")
                    continue

                wfs = raw_data.waveform.values.nda

                if wfs.shape[1] > 0:
                    dts = np.linspace(0, (wfs.shape[1]-1) * 0.008, wfs.shape[1])
                    for j in range(min(10, wfs.shape[0])):
                        wf0 = wfs[j]
                        ax.plot(dts, wf0)
                    ax.set_xlim(0, dts[-1])
                    ax.set_xlabel(r"Time ($\mu$s)")
                    ax.set_ylabel("ADC")
                    ax.set_title(chn)
                else:
                    ax.set_title(f"{chn} (Empty WFs)")

            except Exception as plot_e:
                app_logger.exception(f"Error plotting channel {chn}")
                ax.set_title(f"{chn} (Error)")
                ax.text(0.5, 0.5, "Plot Error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue

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

if __name__ == '__main__':
    if not os.path.exists('daq_scope.py'):
        app_logger.critical("Error: 'daq_scope.py' not found in the current directory. Please place 'daq_scope.py' alongside 'app.py'.")
        exit(1)

    # Use a logger for Flask's internal messages as well
    logging.getLogger('werkzeug').setLevel(logging.INFO) # Adjust as needed
    app.run(debug=True, port=44500)