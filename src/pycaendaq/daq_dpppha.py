import argparse
import os
import time
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import yaml
import ast
import sys

from lgdo import lh5, Table, Array, WaveformTable, ArrayOfEqualSizedArrays
from caen_felib import lib, device, error


def main():
    par = argparse.ArgumentParser(description="Save digitizer data to LH5. Press Ctrl+C during acquisition to stop manually.")
    arg, st = par.add_argument, "store_true"
    arg("-a", "--dig_address", required=True, help="dig2://caendgtz-usb-66154")
    arg("-c", "--config_file", required=True, help="Configuration file name")
    arg("-o", "--out_file", type=str, help="Base output file name")
    arg("-tt", "--temperature", action=st, help="Save temperature")
    arg("-n", "--n_events", type=int, help="Total number of events to acquire")
    arg("-d", "--duration", type=int, help="Maximum acquisition time in seconds")

    args = vars(par.parse_args())

    dig_address = args["dig_address"]
    if args["out_file"]:
        save_enabled = True
        out_file = args["out_file"]
        base_name = out_file.replace(".lh5", "")
    else:
        save_enabled = False

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

    data_format_dict = config.get("data_format", {})
    gen_settings = config.get("general_settings", {})
    channel_settings = config.get("channel_settings", {})
    all_active_channels = set()

    for group_name, group_dict in channel_settings.items():
        channels_str = group_dict.get("channels")
        is_enabled = group_dict.get("chenable")
        if channels_str is None:
            print(f"WARNING: 'channels' key missing for group '{group_name}'. Skipping.")
            continue
        if not is_enabled:
            print(f"Group '{group_name}' is DISABLED. Skipping channels.")
            continue
        group_channels = []
        if ".." in channels_str:
            try:
                limits = channels_str.split("..")
                if len(limits) != 2:
                    print(f"WARNING: Invalid format '{channels_str}' for '{group_name}'. Expected 'start..end'.")
                    continue
                start_channel = int(limits[0])
                end_channel = int(limits[1])
                if start_channel > end_channel:
                    print(f"WARNING: Invalid range '{channels_str}' for '{group_name}'. Start is greater than end.")
                    continue
                group_channels = list(range(start_channel, end_channel + 1))
            except ValueError:
                print(f"ERROR: Could not parse '{channels_str}' for '{group_name}'. Ensure start and end are integers.")
                continue
        else:
            try:
                single_channel = int(channels_str)
                group_channels = [single_channel]
            except ValueError:
                print(f"ERROR: Could not parse channel '{channels_str}' for '{group_name}'. Ensure is integer.")
                continue
        group_dict["channel_list"] = group_channels
        all_active_channels.update(group_channels)

    channel_list = sorted(list(all_active_channels))
    active_ch_count = len(channel_list)

    print(f"Total active channels: {active_ch_count}")
    print(f"Active channels list: {channel_list}")

    max_file_size_mb = gen_settings.get("max_file_size_mb", 100)
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    buffer_size = gen_settings.get("buffer_size", 100)
    interval_stats = gen_settings.get("interval_stats", 100)
    software_rate = gen_settings.get("software_trigger_rate", 1000)

    save_temperature = args["temperature"]
    total_events = args.get("n_events")
    max_duration = args.get("duration")

    if total_events is not None and buffer_size > total_events:
        buffer_size = total_events

    temp_names = [
        "tempsensfirstadc", "tempsenshottestadc", "tempsenslastadc",
        "tempsensairin", "tempsensairout", "tempsenscore", "tempsensdcdc"
    ]

    if save_enabled:
        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        current_file = get_new_filename(base_name, timestamp_str)

    buffer_counter = 0
    start_time = time.time()
    start_timestamp = time.time_ns()

    recordlengths = 1000
    data_format = [
        {
            'name': 'CHANNEL',
            'type': 'U8',
            'dim' : 0
        },
        {
            'name': 'TIMESTAMP',
            'type': 'U64',
            'dim': 0,
        },
        {
            'name': 'ENERGY',
            'type': 'U16',
            'dim': 0,
        },
        {
            'name': 'ANALOG_PROBE_1',
            'type': 'U16',
            'dim': 1,
            'shape': [recordlengths]
        },
        {
            'name': 'ANALOG_PROBE_1_TYPE',
            'type': 'I32',
            'dim': 0
        },
        {
            'name': 'DIGITAL_PROBE_1',
            'type': 'U8',
            'dim': 1,
            'shape': [recordlengths]
        },
        {
            'name': 'DIGITAL_PROBE_1_TYPE',
            'type': 'I32',
            'dim': 0
        },
        {
            'name': 'WAVEFORM_SIZE',
            'type': 'SIZE_T',
            'dim': 0
        }
    ]

    with device.connect(dig_address) as dig:
        dig.cmd.Reset()

        print_dig_stats(dig)

        print(f"--- Applying general digitizer settings ---")
        for param_name, param_value in gen_settings.items():
            if param_name in ["software_trigger_rate","max_file_size_mb",
                              "buffer_size","interval_stats"]:
                continue
            if param_value is None:
                continue
            print(f"  Setting {param_name} to {param_value}")
            dig.set_value(f"/par/{param_name}",str(param_value))

        for ch in dig.ch:
            ch.par.chenable.value = "FALSE"
        for group_name, group_dict in channel_settings.items():
            if "channels" not in group_dict:
                print(f"WARNING: Group '{group_name}' is missing or has an invalid 'channels'.")
                continue
            chns = group_dict["channels"]
            print(f"--- Applying settings for channels {chns} of group '{group_name}' ---")
            for param_name, param_value in group_dict.items():
                if param_name in ["channel_list", "channels"]:
                    continue
                print(f"  /ch/{chns}: Setting {param_name} to {param_value}")
                dig.set_value(f"/ch/{chns}/par/{param_name}",str(param_value))

        tot_channels = int(dig.par.numch.value)
        sampling_period_ns = int(1e3 / float(dig.par.adc_samplrate.value))
        data_format = [
            {"name": "TIMESTAMP_NS", "type": "U64"},
            {"name": "TRIGGER_ID", "type": "U32"},
            {"name": "WAVEFORM", "type": "U16", "dim": 2, "shape": [tot_channels, recordlengths]},
        ]
        
        decoded_endpoint_path = "dpppha"
        endpoint = dig.endpoint[decoded_endpoint_path]
        data = endpoint.set_read_data_format(data_format)
        dig.endpoint.par.activeendpoint.value = decoded_endpoint_path

        channel = data[0].value
        timestamp = data[1].value
        energy = data[2].value
        analog_probe_1 = data[3].value
        analog_probe_1_type = data[4].value  # Integer value described in Supported Endpoints > Probe type meaning
        digital_probe_1 = data[5].value
        digital_probe_1_type = data[6].value  # Integer value described in Supported Endpoints > Probe type meaning
        waveform_size = data[7].value

        print("Arming")
        dig.cmd.armacquisition()
        print("Start acquisition")
        dig.cmd.swstartacquisition()

        n_last = nev % n_split
        n_iter = int(nev/n_split)
        print("Total iterations",n_iter,"last iteration",n_last)

        for n in range(n_iter + 1):
            print("Starting iteration",n)
            if n == n_iter:
                n_current = n_last
            else:
                n_current = n_split
            if n_current == 0: continue

            timestamps = np.zeros((active_ch,n_current),dtype=np.uint64)
            wfs = np.zeros((active_ch,n_current,recordlengths),dtype=np.uint16)

            if save_temperature:
                temp_names = ["tempsensfirstadc","tempsenshottestadc","tempsenslastadc","tempsensairin","tempsensairout","tempsenscore","tempsensdcdc"]
                temperatures = np.zeros((n_current,len(temp_names)),dtype=float)

            t_start = time.time()
            for i in range(n_current*active_ch):
                acq_time = time.time() - t_start
                if (i//active_ch % 1000 == 0 ):
                    print(f"Acquisition of event n. {i}, elapsed time {acq_time:.2f} s")
                try:
                    endpoint.read_data(-1, data)
                    wfs[channel,i//active_ch] = analog_probe_1
                    timestamps[channel,i//active_ch] = timestamp
                    if save_temperature:
                        for j, temp in enumerate(temp_names):
                            temp_value = float(dig.get_value(f"/par/{temp}"))
                            temperatures[i//active_ch][j] = temp_value
                except error.Error as ex:
                    if ex.code == error.ErrorCode.TIMEOUT:
                        continue
                    if ex.code == error.ErrorCode.STOP:
                        break
                    else:
                        raise ex
            
            dt = Array(
                sampling_period_ns * np.ones(n_current, dtype=np.uint16),
                attrs={'datatype': 'array<1>{real}', 'units': 'ns'}
            )
            t0 = Array(
                sampling_period_ns * np.ones(n_current, dtype=np.uint16),
                attrs={'datatype': 'array<1>{real}', 'units': 'ns'},
            )

            for ch in range(active_ch):
                values = ArrayOfEqualSizedArrays(
                    nda=wfs[ch],
                    attrs={'datatype': 'array_of_equalsized_arrays<1,1>{real}', 'units': 'ADC'},
                )
                waveform = WaveformTable(
                    size = n_current,
                    t0 = t0,
                    t0_units = "ns",
                    dt = dt,
                    dt_units = "ns",
                    values = values,
                    values_units = "ADC",
                    attrs = {"datatype":"table{t0,dt,values}"}
                )
                ts = Array(timestamps[ch],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})

                raw_data = Table(col_dict={
                    "waveform": waveform,
                    "timestamp": ts
                })
    
                print(f"Saving channel ch{ch} in lh5 file")
                lh5.write(
                    raw_data,
                    name="raw",
                    lh5_file=f"{out_file.split('.lh5')[0]}_{n:03}.lh5",
                    wo_mode="overwrite",
                    group=f"ch{ch}"
                )
            
            if save_temperature:
                temp0 = Array(temperatures[:,0],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp1 = Array(temperatures[:,1],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp2 = Array(temperatures[:,2],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp3 = Array(temperatures[:,3],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp4 = Array(temperatures[:,4],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp5 = Array(temperatures[:,5],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
                temp6 = Array(temperatures[:,6],attrs={'datatype': 'array<1>{real}', 'units': 'ADC'})
    
                raw_data = Table(col_dict={
                    "temp0": temp0,
                    "temp1": temp1,
                    "temp2": temp2,
                    "temp3": temp3,
                    "temp4": temp4,
                    "temp5": temp5,
                    "temp6": temp6
                })
                print(f"Saving temperatures in lh5 file")
                lh5.write(
                    raw_data,
                    name="raw",
                    lh5_file=f"{out_file.split('.lh5')[0]}_{n:03}.lh5",
                    wo_mode="overwrite",
                    group="dig"
                )
                
        dig.cmd.disarmacquisition()
        print("Disarming")

def get_new_filename(base_name, timestamp_str):
    return f"{base_name}_{timestamp_str}.lh5"

def print_dig_stats(dig):
    modelname = dig.par.modelname.value
    fw_type = dig.par.fwtype.value
    fw_ver = dig.par.fpga_fwver.value
    adc_samplrate_msps = float(dig.par.adc_samplrate.value)  # in Msps
    adc_n_bits = int(dig.par.adc_nbit.value)
    sampling_period_ns = int(1e3 / adc_samplrate_msps)
    inputrange = dig.par.inputrange.value
    inputtype = dig.par.inputtype.value
    print(f"\nDigitizer model: {modelname}, Firmware: {fw_type} v. {fw_ver}")
    print(f"Sampling rate = {adc_samplrate_msps} MHz\n",
          f"n. bit = {adc_n_bits}\n",
          f"Sampling period = {sampling_period_ns} ns\n",
          f"Input dynamic range = {inputrange} V\n",
          f"Input type = {inputtype}\n")

def daq_dpp():
    """
    Entry point for the console script 'daq-dpppha'.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nAcquisition stopped manually by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    daq_dpp()