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
    arg("-a", "--dig_address", required=True, help="dig2://caendgtz-usb-52696")
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

    recordlengths = int(gen_settings["recordlengths"])
    acqtriggersource = gen_settings["acqtriggersource"]

    buffer_counter = 0
    start_time = time.time()
    start_timestamp = time.time_ns()

    with device.connect(dig_address) as dig:
        dig.cmd.reset()

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
        endpoint = dig.endpoint["scope"]
        data = endpoint.set_read_data_format(data_format)
        dig.endpoint.par.activeendpoint.value = "scope"

        dig.cmd.armacquisition()
        dig.cmd.swstartacquisition()

        temperature_buffer = []
        waveform_buffer = np.full((active_ch_count, buffer_size, recordlengths), 0, dtype=np.uint16)
        timestamp_buffer = np.full((active_ch_count, buffer_size), 0, dtype=np.uint64)
        ev_number_buffer = np.full((active_ch_count, buffer_size), 0, dtype=np.uint16)

        print("\nStarting acquisition...")
        while True:
            if "SwTrg" in acqtriggersource:
                #time.sleep(1 / software_rate)
                dig.cmd.sendswtrigger()

            try:
                endpoint.read_data(1000, data)
            except error.Error as ex:
                if ex.code is error.ErrorCode.TIMEOUT:
                    continue
                if ex.code is error.ErrorCode.STOP:
                    break
                raise ex

            timestamp = data[0].value
            trigger_id = data[1].value
            waveform = data[2].value

            if waveform.shape[0] < active_ch_count or waveform.shape[1] != recordlengths:
                print(f"[WARNING] Invalid waveform shape: {waveform.shape} (expected {active_ch_count} x {recordlengths})")
                continue
            
            for i, ch in enumerate(channel_list):
                waveform_buffer[i, buffer_counter, :] = waveform[ch]
                timestamp_buffer[i, buffer_counter] = np.uint64(start_timestamp+timestamp)
                ev_number_buffer[i, buffer_counter] = trigger_id

            if save_temperature:
                temp_values = [float(dig.get_value(f"/par/{name}")) for name in temp_names]
                temperature_buffer.append(temp_values)

            buffer_counter += 1
            
            if (trigger_id % interval_stats) == 0:
                print_stats(dig, start_time, trigger_id)

            if buffer_counter >= buffer_size:
                if save_enabled:
                    print(f"...writing current file: {current_file}, total events {trigger_id}")
                    for i, ch in enumerate(channel_list):
                        if waveform_buffer[i].ndim != 2 or waveform_buffer[i].shape[1] != recordlengths:
                            print(f"[ERROR] Buffer shape mismatch: {waveform_buffer[i].shape}")
                            continue
    
                        values = ArrayOfEqualSizedArrays(
                            nda=waveform_buffer[i],
                            attrs={"datatype": "array_of_equalsized_arrays<1,1>{real}", "units": "ADC"},
                        )
                        wf = WaveformTable(
                            size=buffer_size,
                            t0=Array([0]*buffer_size, attrs={"datatype": "array<1>{real}", "units": "ns"}),
                            dt=Array([sampling_period_ns]*buffer_size, attrs={"datatype": "array<1>{real}", "units": "ns"}),
                            values=values,
                            values_units="ADC"
                        )
    
                        ts_arr = Array(timestamp_buffer[i], attrs={"datatype": "array<1>{real}", "units": "ns"})
                        ev_arr = Array(ev_number_buffer[i], attrs={"datatype": "array<1>{real}"})
                        raw_data = Table(
                            col_dict={"eventnumber":ev_arr, "timestamp": ts_arr, "waveform": wf}
                        )
    
                        lh5.write(raw_data, name="raw", lh5_file=current_file, wo_mode="append", group=f"ch{ch:03}")
    
                    if save_temperature and temperature_buffer:
                        temp_arrs = {
                            f"temp{i}": Array(np.array([row[i] for row in temperature_buffer]),
                                               attrs={"datatype": "array<1>{real}", "units": "C"})
                            for i in range(len(temp_names))
                        }
                        temp_data = Table(col_dict=temp_arrs)
                        lh5.write(temp_data, name="raw", lh5_file=current_file, wo_mode="append", group="dig")
                        temperature_buffer.clear()
    
                    if os.path.exists(current_file) and os.path.getsize(current_file) >= max_file_size_bytes:
                        print(f"File {current_file} exceeded size. Rotating.")
                        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
                        current_file = get_new_filename(base_name, timestamp_str)

                buffer_counter = 0

            if total_events and trigger_id >= total_events:
                print("Reached target number of events. Stopping.")
                print_stats(dig, start_time, trigger_id)
                break
            if max_duration and (time.time() - start_time) >= max_duration:
                print("Reached max acquisition time. Stopping.")
                print_stats(dig, start_time, trigger_id)
                break

        dig.cmd.disarmacquisition()
        print("Acquisition stopped.")

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

def print_stats(dig, start_time, counter):
    elapsed = time.time() - start_time
    rate = counter / elapsed / (1024. * 1024.) # MB/s
    print(f"Elapsed time {elapsed:.1f} s, n. events: {counter}, readout rate {rate:.1e} MB/s")

    status = decode_status(dig.get_value("/par/acquisitionstatus"))
    print(f"Acquisition Status:")
    for key, value in status.items():
        print(f"  {key:17}: {'ON' if value else 'OFF'}")

    par_stats = ["livetimemonitor","realtimemonitor","deadtimemonitor","triggercnt","losttriggercnt"]
    for par in par_stats:
        val = dig.get_value(f'/par/{par}')
        if par in ["livetimemonitor", "realtimemonitor", "deadtimemonitor"]:
            name = par.split("monitor")[0]
            val = int(val)
            val = int(val / 524288)
        else:
            name = par
        print(f"{name}: {val} ", end='')
    print()

def decode_status(status):
    flags = {
        0: "Armed",
        1: "Run",
        2: "Run_mw",
        3: "Jesd_Clk_Valid",
        4: "Busy",
        5: "PreTriggerReady",
        6: "LicenseFail",
    }
    status = int(status)
    decoded_status = {}
    for bit, name in flags.items():
        decoded_status[name] = bool((status >> bit) & 1)

    return decoded_status

if __name__ == "__main__":
    main()
