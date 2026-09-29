from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

import re
import regex

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    # pass
    bench.workload.synchronize() 

    elapsed_times = []
    for i in range(repeats):
        start_time = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end_time = bench.clock()

        elapsed = (end_time - start_time)/1000000.0

        elapsed_times.append(elapsed)

    return elapsed_times
    

def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    # value of src in unknown doesn't matter, whats important is the error message so you know what to fix

    samples_length = len(samples)
    if samples_length < 4:
        return unknown("samples_length", "too few samples in find_warmup_boundary")
    else:
       samples_second_half_start = int(samples_length/2)
       samples_median = statistics.median(samples[samples_second_half_start:samples_length])

       if samples_median <= 0:
            return unknown("samples_median", "samples median <= 0")
       else:
            threshold = samples_median * (1 + WARMUP_TOL)
            greater_count = 0
            for i in range(samples_length):
               if samples[i] > threshold: # ignore "consecutively" keyword in lab instructions
                   greater_count += 1

            return {
                "value": greater_count,
                "source": "leading prefix above (1 + 0.5) x median of the run's second half",
                "status": "ok",
                "settled_rate_ms": round(samples_median, 4),
                "threshold_ms": round(threshold, 4),
                "tolerance": WARMUP_TOL,
                "retained": samples_length - greater_count
                }


def summarize(samples: list[float]) -> dict[str, Any]:
    import numpy as np
    # pass
    if not samples:
        return {
            "n": 0, 
            "mean": None , 
            "std": None, 
            "min": None, 
            "max": None, 
            "p50": None, 
            "p95": None, 
            "p99": None
            }
    else:
        samples.sort()
        s_mean = round(statistics.fmean(samples), 4)
        s_min = round(min(samples), 4)
        s_max = round(max(samples), 4)

        if len(samples) > 2:
            stdv = round(statistics.stdev(samples), 4)
        else:
            stdv = 0.0

        p50 = round(np.percentile(samples, 50), 4)
        p95 = round(np.percentile(samples, 95), 4)
        p99 = round(np.percentile(samples, 99), 4)

        return {
            "n": len(samples), 
            "mean": s_mean , 
            "std": stdv, 
            "min": s_min, 
            "max": s_max, 
            "p50": p50, 
            "p95": p95, 
            "p99": p99
            }


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    # pass
    orig_samples_length = len(samples)
    samples_length = len(samples)

    orig_samples = samples

    if samples_length < 20:
        return unknown("samples", "not enough samples in is_multimodal")
    else:
        samples.sort()
        trunc_val = int(samples_length * 0.05)
        samples = samples[trunc_val:samples_length - trunc_val] 
        # trim: including values starting with (i = trunc_val) to the value right before (samples_length - trunc_val)
        samples_length = len(samples)
        gaps = {}

        for i in range(samples_length - 1):
            gaps[i] = samples[i+1] - samples[i]
        
        median_gap = statistics.median(list(gaps.values()))
        if median_gap <= 0:
            return unknown("median_gap", "timer resolution is too coarse in is_multimodal")

        widest_gap = max(list(gaps.values()))

        ratio = round(widest_gap/median_gap, 2)

        split = trunc_val + list(gaps.keys())[list(gaps.values()).index(widest_gap)]
        left_split = split + 1 
        right_split = orig_samples_length - left_split

        modes = [{
                "n": left_split,
                "share": round(left_split/orig_samples_length, 4),
                "median_ms": round(statistics.median(orig_samples[0:left_split]), 4)
            },
            {
                "n": right_split,
                "share": round(right_split/orig_samples_length, 4),
                "median_ms": round(statistics.median(orig_samples[left_split:orig_samples_length]), 4)
            }]

        measured = {
            "value": False, 
            "source": "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side", 
            "status": "ok", 
            "gap_ratio": ratio, 
            "widest_gap_ms": round(widest_gap, 4), 
            "typical_gap_ms": round(median_gap, 5), 
            "modes": modes, 
            }

        if ratio >= 20.0 and (left_split >= trunc_val and right_split >= trunc_val):
            measured["value"] = True

        return measured


# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    # pass
    nvpmodel_output = bench.runner(["nvpmodel", "-q"])

    if not nvpmodel_output:
        return unknown("nvpmodel -q", "missing sudo permissions in probe_power_state")
    else:

        power_mode = re.search(r"^NV\sPower\sMode:\s(\d+)W\n(\d+)\n", nvpmodel_output.stdout, re.MULTILINE)

        if power_mode:

            mode_name = power_mode.group(1)
            mode_id = power_mode.group(2)

            scaling_min_freq = read_text(bench.telemetry, "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq")
            scaling_max_freq = read_text(bench.telemetry, "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")

            if  scaling_min_freq and scaling_max_freq:
                if scaling_min_freq == scaling_max_freq:
                    jetson_clocks = True
                else:
                    jetson_clocks = False

                return {
                    "value": f"{mode_name}W",
                    "source": "nvpmodel -q",
                    "status": "ok",
                    "mode_index": int(mode_id),
                    "jetson_clocks": jetson_clocks,
                    "jetson_clocks_source": {
                        "value": f"scaling_min_freq={scaling_min_freq}, scaling_max_freq={scaling_max_freq}",
                        "source": "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq vs sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
                        "status": "ok"
                        }
                    }
            
        return unknown("nvpmodel -q", "scaling_min_freq or scaling_max_freq is None in probe_power_state")


def probe_telemetry(bench: Bench) -> dict[str, Any]:
    ## Temperature: ##
    base = bench.telemetry / "sys/devices/virtual/thermal"
    # print(base)
    zones_lst = []
    for zone in base.glob("thermal_zone*"):
        # print(zone)
        # matches any file that has the above string in its path

        try:
            z_temp = read_text(bench.telemetry, f"{zone}/temp")
            # print(z_temp)
        except TypeError: # ignoring type errors but need try-except for code to work
            continue

        if z_temp:
            z_temp = float(z_temp)/1000.0
            if z_temp > -1000:
                zones_lst.append(z_temp)
        else:
            return unknown("sys/devices/virtual/thermal/*/temp", "z_temp is None in probe_telemetry")

    # print(zones_lst)
    max_temp = max(zones_lst)

    temperature = {
        "value": max_temp,
        "source": "sys/devices/virtual/thermal/*/temp",
        "status": "ok",
        "zone": "soc1-thermal",
        "zones_read": len(zones_lst)
        }

    ## Power Draw ##
    power_mw = read_first(bench.telemetry, 
                          ("sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input", 
                           "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input", 
                           "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input"))

    if not power_mw:
        power_mw = unknown("sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input | sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input | sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input", "none of the documented INA3221 rail paths could be read in probe_telemetry")

    ## GPU Load ##
    raw = read_text(bench.telemetry, "sys/devices/platform/gpu.0/load")
    if raw:
        gpu_load = float(raw)/10.0
    else:
        gpu_load = unknown("sys/devices/platform/gpu.0/load", "no GPU load file is found in probe_telemetry")

    gpu_utilization_percent = {
        "value": gpu_load,
        "source": "sys/devices/platform/gpu.0/load",
        "status": "ok",
        "units": "per-mille / 10"
        }
    
    ret = {
        "temperature_c": temperature,
        "power_mw": power_mw,
        "gpu_utilization_percent": gpu_utilization_percent
        }
    
    return ret

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#     # calling base environment
#     env = Bench.real()
#     # get your samples
#     samples = run_timed_iterations(env, repeats=100)
    
#     # out = is_multimodal(samples)
#     out = probe_telemetry(env)
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = [
        463.399936,
        19.887707,
        19.013559,
        18.800366,
        18.818862,
        18.803278,
        18.739435,
        18.816014,
        18.878256,
        19.11097,
        20.59564,
        19.140859,
        18.838319,
        18.842159,
        18.760172,
        18.748684,
        18.670696,
        18.801134,
        18.835855,
        18.803662,
        18.786797,
        18.77102,
        18.807885,
        19.06572,
        18.908818,
        18.865968,
        18.785549,
        18.846224,
        18.801037,
        18.769356,
        19.080761,
        19.014455,
        18.776237,
        18.728906,
        18.846095,
        18.794254,
        18.971124,
        19.023511,
        19.400935,
        19.104474,
        18.989589,
        18.745867,
        18.845295,
        18.825071,
        18.864816,
        18.851728,
        18.841295,
        18.768301,
        19.044536,
        18.927474,
        19.018006,
        18.962645,
        18.811342,
        18.817582,
        19.068665,
        18.989238,
        18.966676,
        18.986934,
        19.050072,
        18.754155,
        18.726762,
        18.668392,
        18.742315,
        18.811694,
        18.740747,
        18.959444,
        18.927058,
        19.030936,
        18.995477,
        18.9629,
        18.783373,
        18.771148,
        18.824495,
        18.712682,
        19.043512,
        19.008342,
        18.714378,
        18.866065,
        18.801966,
        18.768396,
        18.742987,
        18.827695,
        18.869712,
        18.801101,
        18.867441,
        18.872848,
        18.877073,
        18.735499,
        18.816238,
        18.775916,
        18.800622,
        18.816718,
        18.886706,
        18.738411,
        18.834351,
        18.702633,
        18.734154,
        18.843472,
        18.798253,
        18.765453
    ]
    
    # samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "ama_samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "ama_system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)