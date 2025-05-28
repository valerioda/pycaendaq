import argparse
import os
import time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import yaml

from lgdo import lh5, Table, Array, WaveformTable, ArrayOfEqualSizedArrays
from caen_felib import lib, device, error

def get_new_filename(base_name, timestamp_str):
    return f"{base_name}_{timestamp_str}.lh5"

def main():
    par = argparse.ArgumentParser(description="Save digitizer data to LH5. Press Ctrl+C during acquisition to stop manually.")
    arg, st = par.add_argument, "store_true"
    arg("-a", "--dig_address", required=True, help="dig2://caendgtz-usb-52696")
    arg("-o", "--out_file", required=True, help="Base output file name")
    arg("-c", "--config_file", required=True, help="Configuration file name")
    arg("-tt", "--temperature", action=st, help="Save temperature")
    arg("-st", "--software_trigger", action=st, help="Use software trigger")
    arg("-n", "--n_events", type=int, help="Total number of events to acquire")
    arg("-d", "--duration", type=int, help="Maximum acquisition time in seconds")

    args = vars(par.parse_args())

    dig_address = args["dig_address"]
    out_file = args["out_file"]
    base_name = out_file.replace(".lh5", "")

    config = {}
    config_file = args["config_file"]
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_file}")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        print("Please ensure 'config.yaml' is in the same directory as the script, or provide the full path.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing '{config_file}': {exc}")
        print("Please check your YAML file for syntax errors.")
        sys.exit(1)

    gen_settings = config.get("general_settings", {})

    record_length = gen_settings.get("record_length", 4084)
    pretrigger = gen_settings.get("pretrigger", 2042)
    dc_offset = gen_settings.get("dc_offset", "10")
    active_ch = gen_settings.get("n_channels", 1)
    max_file_size_mb = gen_settings.get("max_file_size_mb", 100)
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    buffer_size = gen_settings.get("buffer_size", 100)
    software_rate = gen_settings.get("software_trigger_rate", 1000)

    save_temperature = args["temperature"]
    trig_source = "SwTrg" if args["software_trigger"] else "TrgIn"
    total_events = args.get("n_events")
    max_duration = args.get("duration")

    temp_names = [
        "tempsensfirstadc", "tempsenshottestadc", "tempsenslastadc",
        "tempsensairin", "tempsensairout", "tempsenscore", "tempsensdcdc"
    ]

    timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    current_file = get_new_filename(base_name, timestamp_str)

    data_format = [
        {"name": "EVENT_SIZE", "type": "SIZE_T"},
        {"name": "TIMESTAMP", "type": "U64"},
        {"name": "WAVEFORM", "type": "U16", "dim": 2, "shape": [active_ch, record_length]},
        {"name": "WAVEFORM_SIZE", "type": "U64", "dim": 1, "shape": [active_ch]},
    ]

    waveform_buffer = [[] for _ in range(active_ch)]
    timestamp_buffer = [[] for _ in range(active_ch)]
    temperature_buffer = []
    event_counter = 0
    start_time = time.time()

    with device.connect(dig_address) as dig:
        dig.cmd.reset()

        fw_type = dig.par.fwtype.value
        fw_ver = dig.par.fpga_fwver.value
        print("Firmware",fw_type, fw_ver)

        n_ch = int(dig.par.numch.value)
        adc_samplrate_msps = float(dig.par.adc_samplrate.value)  # in Msps
        adc_n_bits = int(dig.par.adc_nbit.value)
        sampling_period_ns = int(1e3 / adc_samplrate_msps)
        
        #print(f"Sampling rate = {adc_samplrate_msps} MHz, n. bit = {adc_n_bits}, Sampling period = {sampling_period_ns} ns")

        nch = int(dig.par.NumCh.value)
        dig.par.iolevel.value = "TTL"
        dig.par.acqtriggersource.value = trig_source
        dig.par.recordlengths.value = f"{record_length}"
        dig.par.pretriggers.value = f"{pretrigger}"

        for i, ch in enumerate(dig.ch):
            ch.par.chenable.value = "TRUE" if i < active_ch else "FALSE"
            ch.par.dcoffset.value = dc_offset

        endpoint = dig.endpoint["scope"]
        data = endpoint.set_read_data_format(data_format)
        dig.endpoint.par.activeendpoint.value = "scope"

        dig.cmd.armacquisition()
        dig.cmd.swstartacquisition()

        print("\nStarting acquisition...")
        try:
            while True:
                if trig_source == "SwTrg":
                    time.sleep(1 / software_rate)
                    dig.cmd.sendswtrigger()

                try:
                    endpoint.read_data(100, data)
                except error.Error as ex:
                    if ex.code is error.ErrorCode.TIMEOUT:
                        continue
                    if ex.code is error.ErrorCode.STOP:
                        break
                    raise ex

                waveform = data[2].value
                timestamp = data[1].value

                for ch in range(active_ch):
                    waveform_buffer[ch].append(waveform[ch])
                    timestamp_buffer[ch].append(np.uint64(timestamp))

                if save_temperature:
                    temp_values = [float(dig.get_value(f"/par/{name}")) for name in temp_names]
                    temperature_buffer.append(temp_values)

                event_counter += 1

                # Check for stopping conditions
                if total_events and event_counter >= total_events:
                    print("Reached target number of events. Stopping.")
                    break
                if max_duration and (time.time() - start_time) >= max_duration:
                    print("Reached max acquisition time. Stopping.")
                    break

                if len(timestamp_buffer[0]) >= buffer_size:
                    print(f"...writing current file: {current_file}, total events {event_counter}")
                    for ch in range(active_ch):
                        wfs = np.array(waveform_buffer[ch], dtype=np.uint16)
                        ts = np.array(timestamp_buffer[ch], dtype=np.uint64)

                        values = ArrayOfEqualSizedArrays(
                            nda=wfs,
                            attrs={"datatype": "array_of_equalsized_arrays<1,1>{real}", "units": "ADC"},
                        )
                        wf = WaveformTable(size=len(ts),
                                           t0=Array([0]*len(ts), attrs={"datatype": "array<1>{real}", "units": "ns"}),
                                           dt=Array([1]*len(ts), attrs={"datatype": "array<1>{real}", "units": "ns"}),
                                           values=values,
                                           values_units="ADC")

                        ts_arr = Array(ts, attrs={"datatype": "array<1>{real}", "units": "ADC"})
                        raw_data = Table(col_dict={"waveform": wf, "timestamp": ts_arr})

                        #lh5.write(raw_data, name="raw", lh5_file=current_file, wo_mode="append", group=f"ch{ch:03}")

                        waveform_buffer[ch].clear()
                        timestamp_buffer[ch].clear()

                    if save_temperature and temperature_buffer:
                        temp_arrs = {
                            f"temp{i}": Array(np.array([row[i] for row in temperature_buffer]),
                                               attrs={"datatype": "array<1>{real}", "units": "C"})
                            for i in range(len(temp_names))
                        }
                        temp_data = Table(col_dict=temp_arrs)
                        lh5.write(temp_data, name="raw", lh5_file=current_file, wo_mode="append", group="dig")
                        temperature_buffer.clear()

                    if os.path.exists(current_file) and os.path.getsize(current_file) >= max_file_size:
                        print(f"File {current_file} exceeded size. Rotating.")
                        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
                        current_file = get_new_filename(base_name, timestamp_str)

        except KeyboardInterrupt:
            #dig.cmd.swstopacquisition()
            print("KeyboardInterrupt received. Stopping acquisition.")

        dig.cmd.disarmacquisition()
        print("Acquisition stopped.")

if __name__ == "__main__":
    main()
