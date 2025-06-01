import argparse
import os
import time
import numpy as np
from matplotlib import pyplot as plt

from lgdo import lh5, Table, Array, WaveformTable, ArrayOfEqualSizedArrays
from caen_felib import lib, device, error

dig2_scheme = "dig2"
#dig2_authority = "caendgtz-usb-21233"
dig2_authority = "caendgtz-usb-52696"
dig2_query = ''
dig2_path = ''
dig2_uri = f'{dig2_scheme}://{dig2_authority}/{dig2_path}?{dig2_query}'


def main():
    par = argparse.ArgumentParser(description="save hit energies")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-n", "--n_events", nargs=1, help="n_events")
    arg("-o", "--out_file", nargs=1, help="out_file")
    arg("-nc", "--n_ch", nargs=1, help="n_ch")
    arg("-rl", "--record_length", nargs=1, help="record length in samples")
    arg("-pt", "--pretrigger", nargs=1, help="pre trigger in samples")
    arg("-dc", "--dc_offset", nargs=1, help="dc_offset in percentage")
    arg("-tt", "--temperature", action=st, help="temperature")
    args = vars(par.parse_args())

    if args["n_events"]:
        nev = int(args['n_events'][0])
    else:
        print("Number of events not provided")
        return
    
    if args["out_file"]:
        out_file = args['out_file'][0]
    else:
        print("Output file not provided")
        return

    if args["record_length"]:
        reclen = int(args['record_length'][0])
    else:
        reclen = 8000

    if args["pretrigger"]:
        pretrg = int(args['record_length'][0])
    else:
        pretrg = 4000

    if args["dc_offset"]:
        dc_offset = args['dc_offset'][0]
    else:
        dc_offset = f"{10}"

    if args["temperature"]:
        save_temperature = True
    else:
        save_temperature = False
    
    if args["n_ch"]:
        active_ch = int(args["n_ch"][0])
    else:
        active_ch = 1

    n_split = 10000

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
            'shape': [reclen]
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
            'shape': [reclen]
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

    with device.connect(dig2_uri) as dig:
        
        dig.cmd.Reset()

        fw_type = dig.par.fwtype.value
        fw_ver = dig.par.fpga_fwver.value
        print("Firmware",fw_type, fw_ver)

        n_ch = int(dig.par.numch.value)
        adc_samplrate_msps = float(dig.par.adc_samplrate.value)  # in Msps
        adc_n_bits = int(dig.par.adc_nbit.value)
        sampling_period_ns = int(1e3 / adc_samplrate_msps)
        
        print(f"Sampling rate = {adc_samplrate_msps} MHz, n. bit = {adc_n_bits}, Sampling period = {sampling_period_ns} ns")

        nch = int(dig.par.NumCh.value)
        dig.par.iolevel.value = "TTL"
        dig.par.globaltriggersource.value = "TrgIn"

        print("Set channel parameters")
        for i, ch in enumerate(dig.ch):
            ch.par.chenable.value = "TRUE" if i < active_ch else "FALSE"
            ch.par.eventtriggersource.value = "GLOBALTRIGGERSOURCE"
            ch.par.wavetriggersource.value = "GLOBALTRIGGERSOURCE"
            ch.par.chrecordlengths.value = f"{reclen}"
            ch.par.chpretriggers.value = f"{pretrg}"
            ch.par.waveanalogprobe0.value = "ADCINPUT"
            ch.par.wavedigitalprobe0.value = "TRIGGER"
            ch.par.dcoffset.value = dc_offset

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
            wfs = np.zeros((active_ch,n_current,reclen),dtype=np.uint16)

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

if __name__ == "__main__":
    main()