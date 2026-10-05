import statistics
from typing import Any
from bench import unknown, Bench
from measure import MIN_SAMPLES_FOR_STATIONARITY, STATIONARITY_TOL, run_timed_iterations
import math

def is_stationary(samples: list[float]) -> dict[str, Any]:
	# *PART 1*
	n = len(samples)
	if n < MIN_SAMPLES_FOR_STATIONARITY: # measure.py
		return unknown("input samples", "too few samples to divide into thirds") # bench.py

	samples_median = statistics.median(samples)
	# print(f"samples_median is {samples_median}")
	if samples_median <= 0:
		return unknown("input samples", "median is not positive") # bench.py

	# *PART 2*
	k = math.floor(n/3)
	# print(f"k is {k}\n")
	samples_median_first_third = statistics.median(samples[0:k])
	# print(f"first_third is {samples[0:k]}, and its length is {len(samples[0:k])}\n")
	# print(f"last_third is {samples[n-k:n]}, and its length is {len(samples[n-k:n])}\n")
	# print(f"second_third length is {len(samples[k:n-k])}\n")
	samples_median_last_third = statistics.median(samples[n-k:n])

	# *PART 3*
	raw_drift = samples_median_last_third - samples_median_first_third
	relative_drift = abs(raw_drift)/samples_median 

	# *PART 4*
	if raw_drift > 0:
		drift_direction = "slower"
	elif raw_drift < 0:
		drift_direction = "faster"
	else:
		drift_direction = "flat"

	# *PART 5*
	measured = {
		"value": False,
		"first_third_median_ms": round(samples_median_first_third, 4),
        "last_third_median_ms": round(samples_median_last_third, 4),
        "drift_ms": round(raw_drift, 4),
        "drift_relative": round(relative_drift, 4),
        "direction": drift_direction,
        "tolerance": STATIONARITY_TOL # measure.py
        }
	

	if relative_drift <= STATIONARITY_TOL: # measure.py
		measured["value"] = True

	return measured

if __name__ == "__main__":
	# env = Bench.real()
	# samples = run_timed_iterations(env, repeats=100)
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
	out = is_stationary(samples)
	print(out)

    # 0 1 2 3 4| 5 6 7 8 9 |10 11 12 13 14 