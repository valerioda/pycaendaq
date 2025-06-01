\\Py-OLAF-DAQ

A Python library for OLAF Data Acquisition (DAQ) systems, including digitizer control and a web interface.
Features
Digitizer Control: Directly interface with DAQ digitizers for data acquisition.
Configurable Acquisition: Supports configuration via YAML files for flexible DAQ settings.
LH5 Data Saving: Saves acquired data in the LH5 format for efficient storage and analysis.
Real-time Monitoring: Command-line output provides live status updates during acquisition.
Web-based Interface: A Flask-based web application for starting/stopping acquisition, viewing logs, and plotting waveforms (including FFTs) from acquired data.
Responsive Plotting: Generates and displays plots of acquired waveforms directly in the web browser.

Installation

To install py-olaf-daq, follow these steps. It's recommended to use a virtual environment.

Clone the repository:
- git clone https://github.com/valerioda/py-olaf-daq.git
- cd py-olaf-daq


Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`


Install build dependencies:
pip install build


Install the package:
For development (editable mode, changes to source code are immediately reflected):
pip install -e .

Or, to build and install a static version:
python -m build
pip install dist/*.whl


Usage
Command-Line Interface (CLI)
The olaf-daq command allows you to control the digitizer directly.
Example:
To start an acquisition using a configuration file, saving to an output file:
olaf-daq -a dig2://caendgtz-usb-52696 -c config.yaml -o /tmp/my_daq_data


For more options, use the --help flag:
olaf-daq --help


Web Interface
The olaf-daq-web command launches the Flask web application, providing a graphical interface for DAQ control and monitoring.
Start the web application:
olaf-daq-web

The application will typically run on http://127.0.0.1:44500/. Open this URL in your web browser.
Using the Web Interface:
Data Acquisition Tab: Configure digitizer address, config file, output file, and acquisition parameters. Start and stop acquisitions. View live logs from the DAQ process.
Waveform Plotting Tab: Plot waveforms (first 10 events, last event, or FFT) from the most recently acquired LH5 file.
Important: The web application relies on daq_scope.py being able to run as a subprocess. Ensure your config.yaml and output directories are accessible from where the olaf-daq-web command is executed.
Project Structure
The project follows a standard src layout:
py-olaf-daq/
├── src/
│   └── py_olaf_daq/        # The actual Python package
│       ├── __init__.py     # Marks as a package, defines __version__
│       ├── daq_scope.py    # Main DAQ control script (CLI entry point)
│       ├── app.py          # Flask web application logic (Web UI entry point)
│       ├── templates/      # HTML templates for the Flask app (e.g., index.html)
│       │   └── index.html
├── pyproject.toml          # Project metadata and build configuration
├── README.md               # This file
├── tests/                  # Unit and integration tests
└── configs/             # Example configuration file


Dependencies
The core dependencies for py-olaf-daq include:
numpy: For numerical operations.
matplotlib: For plotting functionalities.
pyyaml: For reading YAML configuration files.
legend-pydataobj: For handling LH5 data.
caen-felib: For interfacing with CAEN digitizers (ensure this library is correctly installed and configured for your hardware).
Flask: The web framework for the user interface.
scipy: For scientific computing, including signal processing (e.g., FFT).
Dependencies are managed via pyproject.toml and installed automatically with pip.
