# Py OLAF DAQ

A Python library for OLAF Data Acquisition (DAQ) systems, including digitizer control and a web interface.

## Features
- Digitizer Control: Directly interface with DAQ digitizers for data acquisition.
- Configurable Acquisition: Supports configuration via YAML files for flexible DAQ settings.
- LH5 Data Saving: Saves acquired data in the LH5 format for efficient storage and analysis.
- Real-time Monitoring: Command-line output provides live status updates during acquisition.
- Web-based Interface: A Flask-based web application for starting/stopping acquisition, viewing logs, and plotting waveforms (including FFTs) from acquired data.
- Responsive Plotting: Generates and displays plots of acquired waveforms directly in the web browser.

## Installation

To install pyolafdaq, follow these steps. It's recommended to use a virtual environment.

Clone the repository:
```console
git clone git@github.com:valerioda/pyolafdaq.git
cd pyolafdaq
pip install .
```

## Usage
Command-Line Interface (CLI)
The daq-scope command allows you to control the digitizer directly.
Example:
To start an acquisition using a configuration file, saving to an output file:
```console
daq-scope -a dig2://caendgtz-usb-52696 -c configs/config_scope.yaml -o /tmp/my_daq_data
```

For more options, use the --help flag:
```console
daq-scope --help
```

## Web Interface
The daq-web-app command launches the Flask web application, providing a graphical interface for DAQ control and monitoring.
1. Start the web application:
```console
daq-web-app
```
The application will typically run on http://127.0.0.1:44500/. Open this URL in your web browser.

2. Using the Web Interface:
  - Data Acquisition Tab: Configure digitizer address, config file, output file, and acquisition parameters. Start and stop acquisitions. View live logs from the DAQ process.
  - Waveform Plotting Tab: Plot waveforms (first 10 events, last event, or FFT) from the most recently acquired LH5 file.

Important: The web application relies on daq_scope.py being able to run as a subprocess. Ensure your config.yaml and output directories are accessible from where the daq-web-app command is executed.

## Project Structure
The project follows a standard src layout:
```console
pyolafdaq/
├── src/
│   └── pyolafdaq/        # The actual Python package
│       ├── __init__.py     # Marks as a package, defines __version__
│       ├── daq_scope.py    # Main DAQ control script (CLI entry point)
│       ├── app.py          # Flask web application logic (Web UI entry point)
│       ├── templates/      # HTML templates for the Flask app (e.g., index.html)
│       │   └── index.html
├── pyproject.toml          # Project metadata and build configuration
├── README.md               # This file
└── configs/             # Example configuration file
```

## Dependencies

The core dependencies for pyolafdaq include:
- numpy: For numerical operations.
- matplotlib: For plotting functionalities.
- pyyaml: For reading YAML configuration files.
- legend-pydataobj: For handling LH5 data.
- caen-felib: For interfacing with CAEN digitizers (ensure this library is correctly installed and configured for your hardware).
- Flask: The web framework for the user interface.
- scipy: For scientific computing, including signal processing (e.g., FFT).
Dependencies are managed via pyproject.toml and installed automatically with pip.
