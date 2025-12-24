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

    globaltriggersource = gen_settings["globaltriggersource"]
    buffer_counter = 0
    start_time = time.time()
    start_timestamp = time.time_ns()

    with device.connect(dig_address) as dig:
        dig.cmd.Reset()

        print_dig_stats(dig)

        print(f"--- Applying general digitizer settings ---")
        for param_name, param_value in gen_settings.items():
            if param_name in ["max_file_size_mb","buffer_size","interval_stats"]:
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
        maxrawdatasize = int(dig.par.maxrawdatasize.value)
        sampling_period_ns = int(1e3 / float(dig.par.adc_samplrate.value))

        data_format = [
            {"name": "CHANNEL", "type": device.DataType.U8, "dim": 0},
            {"name": "TIMESTAMP", "type": device.DataType.U64, "dim": 0},
            {"name": "TIMESTAMP_NS", "type": device.DataType.DOUBLE, "dim": 0},
            {"name": "FINE_TIMESTAMP", "type": device.DataType.U16, "dim": 0},
            {"name": "ENERGY", "type": device.DataType.U16, "dim": 0},
            {"name": "FLAGS_LOW_PRIORITY", "type": device.DataType.U16, "dim": 0},
            {"name": "FLAGS_HIGH_PRIORITY", "type": device.DataType.U16, "dim": 0},
            {"name": "TRIGGER_THR", "type": device.DataType.U16, "dim": 0},
            {"name": "TIME_RESOLUTION", "type": device.DataType.U8, "dim": 0},
            {"name": "ANALOG_PROBE_1", "type": device.DataType.I32, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "ANALOG_PROBE_1_TYPE", "type": device.DataType.U8, "dim": 0},
            {"name": "ANALOG_PROBE_2", "type": device.DataType.I32, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "ANALOG_PROBE_2_TYPE", "type": device.DataType.U8, "dim": 0},
            {"name": "DIGITAL_PROBE_1", "type": device.DataType.U8, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "DIGITAL_PROBE_1_TYPE", "type": device.DataType.U8, "dim": 0},
            {"name": "DIGITAL_PROBE_2", "type": device.DataType.U8, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "DIGITAL_PROBE_2_TYPE", "type": device.DataType.U8, "dim": 0},
            {"name": "DIGITAL_PROBE_3", "type": device.DataType.U8, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "DIGITAL_PROBE_3_TYPE", "type": device.DataType.U8, "dim": 0},
            {"name": "DIGITAL_PROBE_4", "type": device.DataType.U8, "dim": 1, "shape": [maxrawdatasize]},
            {'name': "DIGITAL_PROBE_4_TYPE", "type": device.DataType.U8, "dim": 0},
        ]
        keys = [
            "channel", "timestamp", "timestamp_ns", "fine_timestamp", "energy",
            "flag_low", "flag_high", "trigger_thr", "time_res"
        ]
        mapping = {name: i for i, name in enumerate(keys)}
        mapping.update({
            "analog_probe_1": 9,
            "analog_probe_2": 11,
            "digital_probe_1": 13,
            "digital_probe_2": 15,
            "digital_probe_3": 17,
            "digital_probe_4": 19,
        })
        
        decoded_endpoint_path = "dpppha"
        endpoint = dig.endpoint[decoded_endpoint_path]
        data = endpoint.set_read_data_format(data_format)
        dig.endpoint.par.activeendpoint.value = decoded_endpoint_path

        dig.cmd.armacquisition()
        dig.cmd.swstartacquisition()

        temperature_buffer = []
        buffers = {ch: {
            "waveform": [],
            "time_filter": [],
            "digital_1": [],
            "digital_2": [],
            "digital_3": [],
            "digital_4": [],
            "timestamp": [],
            "energy": [],
            "flag_low": [],
            "flag_high": [],
            "count": 0
        } for ch in channel_list}

        print("\nStarting acquisition...")
        trigger_id = 0
        while True:
            if "SwTrg" in globaltriggersource:
                dig.cmd.sendswtrigger()

            try:
                endpoint.read_data(-1, data)
            except error.Error as ex:
                if ex.code is error.ErrorCode.TIMEOUT:
                    continue
                if ex.code is error.ErrorCode.STOP:
                    break
                raise ex

            event = {name: data[idx].value.copy() for name, idx in mapping.items()}
            ch = int(event["channel"])
            chrecordlengths = int(dig.get_value(f"/ch/{ch}/par/chrecordlengths"))
            if dig.get_value(f"/ch/{ch}/par/waveanalogprobe0") == "ADCInput16":
                chrecordlengths *= 2
            buffers[ch]["waveform"].append(event["analog_probe_1"][:chrecordlengths])
            buffers[ch]["time_filter"].append(event["analog_probe_2"][:chrecordlengths])
            buffers[ch]["digital_1"].append(event["digital_probe_1"][:chrecordlengths])
            buffers[ch]["digital_2"].append(event["digital_probe_2"][:chrecordlengths])
            buffers[ch]["digital_3"].append(event["digital_probe_3"][:chrecordlengths])
            buffers[ch]["digital_4"].append(event["digital_probe_4"][:chrecordlengths])
            buffers[ch]["timestamp"].append(np.uint64(start_timestamp + event["timestamp_ns"]))
            buffers[ch]["energy"].append(event["energy"])
            buffers[ch]["flag_low"].append(event["flag_low"])
            buffers[ch]["flag_high"].append(event["flag_high"])
            buffers[ch]["count"] += 1

            if save_temperature:
                temp_values = [float(dig.get_value(f"/par/{name}")) for name in temp_names]
                temperature_buffer.append(temp_values)

            buffer_counter += 1
            if (trigger_id % interval_stats) == 0:
                print_stats(dig, start_time, trigger_id, channel_list)

            if buffer_counter > buffer_size:
                if save_enabled:
                    buffers = flush_buffers_to_lh5(
                        buffers, trigger_id, channel_list, current_file, sampling_period_ns
                    )
                    if os.path.exists(current_file) and os.path.getsize(current_file) >= max_file_size_bytes:
                        print(f"File {current_file} exceeded size. Rotating.")
                        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%SZ")
                        current_file = get_new_filename(base_name, timestamp_str)
                buffer_counter = 0
            trigger_id += 1

            if total_events and trigger_id >= total_events:
                print("Reached target number of events. Stopping.")
                print_stats(dig, start_time, trigger_id, channel_list)
                if save_enabled:
                    buffers = flush_buffers_to_lh5(
                        buffers, trigger_id, channel_list, current_file, sampling_period_ns
                    )
                break
            if max_duration and (time.time() - start_time) >= max_duration:
                print("Reached max acquisition time. Stopping.")
                print_stats(dig, start_time, trigger_id, channel_list)
                if save_enabled:
                    buffers = flush_buffers_to_lh5(
                        buffers, trigger_id, channel_list, current_file, sampling_period_ns
                    )
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


def print_stats(dig, start_time, counter, channel_list):
    elapsed = time.time() - start_time
    rate = counter / elapsed / (1024. * 1024.) # MB/s
    print(f"Elapsed time {elapsed:.1f} s, n. events: {counter}, readout rate {rate:.1e} MB/s")

    status = decode_status(dig.get_value("/par/acquisitionstatus"))
    print(f"Acquisition Status:")
    for key, value in status.items():
        print(f"  {key:17}: {'ON' if value else 'OFF'}")

    for ch in channel_list:
        real_time = dig.get_value(f"/ch/{ch}/par/chrealtimemonitor")
        dead_time = dig.get_value(f"/ch/{ch}/par/chdeadtimemonitor")
        trigger_cnt = dig.get_value(f"/ch/{ch}/par/chtriggercnt")
        wave_cnt = dig.get_value(f"/ch/{ch}/par/chwavecnt")
        saved_event_cnt = dig.get_value(f"/ch/{ch}/par/chsavedeventcnt")
        self_trg_rate = dig.get_value(f"/ch/{ch}/par/selftrgrate")
        stats_dict = {
            "real_time": real_time, 
            "dead_time": dead_time,
            "trigger_cnt": trigger_cnt,
            "wave_cnt": wave_cnt,
            "saved_event_cnt": saved_event_cnt,
            "self_trg_rate": self_trg_rate,
        }
        print(f"[STATS] ch{ch:03}: ", end="")
        for par, val in stats_dict.items():
            if "time" in par:
                val = int(val) // 524288
            print(f"{par}: {val} ", end="")
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


def flush_buffers_to_lh5(
    buffers,
    trigger_id,
    channel_list,
    current_file,
    sampling_period_ns,
    save_temperature=False,
    temperature_buffer=None,
    temp_names=None
):
    """
    Converts buffered list data into LH5 Table structures and writes to disk.
    """

    channel_str = " | ".join([f"{ch}: {buffers[ch]['count']}" for ch in channel_list])
    output = f"[FILE] {current_file:<20} [TOTAL] {trigger_id:<8} [CHANNELS] {channel_str}"
    print(output, flush=True)

    for ch in channel_list:
        ch_data = buffers[ch]
        ch_size = len(ch_data["waveform"])

        if ch_size == 0:
            continue

        def make_waveform_table(data_key, units="ADC"):
            values = ArrayOfEqualSizedArrays(
                nda=np.array(ch_data[data_key]),
                attrs={
                    "datatype": "array_of_equalsized_arrays<1,1>{real}", 
                    "units": units
                },
            )
            return WaveformTable(
                size=ch_size,
                t0=Array([0] * ch_size, attrs={"datatype": "array<1>{real}"}),
                dt=Array([sampling_period_ns] * ch_size, attrs={"datatype": "array<1>{real}", "units": "ns"}),
                values=values,
                values_units=units
            )

        def make_lh5_arr(data_key, units=None):
            attrs = {"datatype": "array<1>{real}"}
            if units: attrs["units"] = units
            return Array(ch_data[data_key], attrs=attrs)
    
        raw_data = Table(
            col_dict={
                "eventnumber": Array(np.arange(buffers[ch]["count"]-ch_size,buffers[ch]["count"]), attrs={"datatype": "array<1>{real}"}),
                "timestamp":   make_lh5_arr("timestamp", units="ns"),
                "energy":      make_lh5_arr("energy"),
                "flag_low":    make_lh5_arr("flag_low"),
                "flag_high":   make_lh5_arr("flag_high"),
                "waveform":    make_waveform_table("waveform"),
                "time_filter": make_waveform_table("time_filter"),
                "digital_1":   make_waveform_table("digital_1"),
                "digital_2":   make_waveform_table("digital_2"),
                "digital_3":   make_waveform_table("digital_3"),
                "digital_4":   make_waveform_table("digital_4")
            }
        )
        lh5.write(raw_data, name="raw", lh5_file=current_file, wo_mode="append", group=f"ch{ch:03}")
        for key in buffers[ch]:
            if key in ["count"]: continue
            else: buffers[ch][key].clear()

    if save_temperature and temperature_buffer and temp_names:
        temp_np = np.array(temperature_buffer)
        temp_arrs = {
            f"temp{i}": Array(
                temp_np[:, i], 
                attrs={"datatype": "array<1>{real}", "units": "C"}
            )
            for i in range(len(temp_names))
        }
        temp_data = Table(col_dict=temp_arrs)
        lh5.write(temp_data, name="raw", lh5_file=current_file, wo_mode="append", group="dig")
        temperature_buffer.clear()
    return buffers

        
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