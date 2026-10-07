from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)

import math

FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "conv":
        c_out = ly.out_shape[0]
        c_in = ly.in_shape[0]

        if ly.kernel:
            (k_h, k_w) = ly.kernel

        else:
            (k_h, k_w) = (1, 1)

        n_weights = int(c_out * (c_in/ly.groups) * k_h * k_w)

        if ly.bias:
            return c_out + n_weights
        else:
            return n_weights

    elif ly.kind == "linear":
        f_out = ly.out_shape[0]
        f_in = ly.in_shape[0]

        if ly.bias:
            weights = (f_out * f_in) + f_out
        else:
            weights = (f_out * f_in)

        return weights

    elif ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]

    else:
        return 0

def count_parameters(graph: Graph) -> dict[str, Any]:
    # pass
    per_layer = {}
    
    for ly in graph.layers:
        param_count = _layer_parameters(ly)
        per_layer[ly.name] = param_count

    total = sum(per_layer.values())

    return {
        "value": total,
        "source": f"{graph.name}: {len(graph)} layers, shapes from the description",
        "status": "computed",
        "per_layer": per_layer,
        "includes_bias": True,
        "excludes_bn_buffers": True,
        "bn_params_per_channel": BN_PARAMS_PER_CHANNEL
        }

 
# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    # pass

    per_dtype = {} # mapping datatype names to byte counts
    per_layer = {} # mapping layer names to byte counts
    buffer_bytes = 0.0

    for ly in graph.layers:
        n = _layer_parameters(ly)
        param_bytes = n * dtype_bytes(ly.weight_dtype) 
        # num of bytes = number of layers * weight (byte size) of the layer

        # ly.weight_dtype is a datatype name (fp32, fp16, bf16, int8, or int4)
        
        if ly.weight_dtype in per_dtype:
            per_dtype[ly.weight_dtype] = per_dtype[ly.weight_dtype] + param_bytes
            # the only datatype that the sample graph layers have is fp32 so per_dtype's only key is fp32
        else:
            per_dtype[ly.weight_dtype] = param_bytes

        if ly.kind == "bn": # theres only ONE layer with kind of "bn"

            # **1
            buffer_elements = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]

            # **2
            buffer_bytes = buffer_bytes + (buffer_elements * dtype_bytes(BUFFER_DTYPE))
            # number of buffer byes

            # **3
            if BUFFER_DTYPE in per_dtype:
                per_dtype[BUFFER_DTYPE] = per_dtype[BUFFER_DTYPE] + buffer_bytes
                # multiple layers can have the same d_type weight
            else:
                per_dtype[BUFFER_DTYPE] = buffer_bytes

            per_layer[ly.name] = int(param_bytes + buffer_bytes) # int cast to match sample output

        else:
            per_layer[ly.name] = int(param_bytes) # int cast to match sample output

        # print(per_dtype)

    total = int(sum(per_layer.values()))

    return {
        "value": total,
        "source": f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}",
        "status": "computed",
        "per_layer": per_layer,
        "per_dtype": per_dtype,
        "buffer_bytes": buffer_bytes,
        "container_overhead_excluded": True,
        "note": "not the size of the file on disk; see the handout, Stage A step 3"
        }


# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def _elements(shape: tuple[int, ...]) -> int:
   elements_product = math.prod(shape)
   return elements_product

# tensors are just the name of a layer
def _last_use(graph: Graph) -> dict[str, int]:
    last = {}
    for i in range(len(graph.layers)):
        last.setdefault(graph.layers[i].name) # defaults value for layer's name tensor as None
        last[graph.layers[-1].name] = len(graph) - 1 # keep the final layer’s output alive until the end

        if graph.layers[i].reads: # if layer has reads then include all of that layer's tensors in last
            for tensor in graph.layers[i].reads:
                last[tensor] = i
        else: # if layer DOESNT have reads tensor then only include name tensor in last
            if i == 0:
                last["__input__"] = 0
            if i > 0:
                last[graph.layers[i-1].name] = i

    return last
        
def _peak_elements(graph: Graph, last_use: dict[str, int]) -> int:
    live = {"__input__": _elements(graph.input_shape)}
    peak_elements = 0

    for i in range(len(graph.layers)):
        live[graph.layers[i].name] = graph.layers[i].out_elements
        print(peak_elements)
        peak_elements = max(peak_elements, sum(live.values()))
        print(live.values())
        # if last_use[graph.layers[i].name] == i: # remove any tensor whose last use index matches i
        #     # print(f"last_use[graph.layers[i] is {graph.layers[i].name} and {last_use[graph.layers[i].name]}\n")
        #     live.pop(graph.layers[i].name)

        for tensor in last_use:
            if last_use[tensor] == i:
                # print(tensor)
                live.pop(tensor) # remove ANY tensor whose last use index matches i

        
        # print(live)

    # print(live)
    return int(peak_elements)


def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    # pass

    last_use = _last_use(graph)
    peak_elements = _peak_elements(graph, last_use)

    live = {"__input__": _elements(graph.input_shape) * dtype_bytes(graph.precision)}

    total_elements = 0
    total_bytes = 0
    peak_bytes = 0

    for i in range(len(graph.layers)):
        output_bytes = graph.layers[i].out_elements * dtype_bytes(graph.layers[i].act_dtype)
        # print(f"output_bytes {i}: output_bytes")
        live[graph.layers[i].name] = output_bytes

        total_elements = total_elements + graph.layers[i].out_elements
        total_bytes = float(total_bytes + output_bytes)

        curr_resident_memory = sum(live.values())
        # print(curr_resident_memory)
        if curr_resident_memory > peak_bytes:
            peak_bytes = curr_resident_memory
            peak_at = graph.layers[i].name

        for tensor in last_use:
            if last_use[tensor] == i:
                # print(tensor)
                live.pop(tensor)

    # print(f"\nfunc live is {live}")

    return {
        "value": peak_bytes,
        "source": f"{graph.name}: liveness over {len(graph)} layers, input included",
        "status": "computed",
        "peak_at": peak_at,
        "peak_elements": peak_elements,
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "includes_network_input": True,
        "note": "peak is the resident set, not the largest single tensor"
        }

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    # pass

    if not is_answered(macs):
        return unknown("input macs", "no valid MAC count was provided")

    if convention not in FLOP_CONVENTIONS:
        return unknown("input convention", f"unrecognized convention, {convention}, is not in {FLOP_CONVENTIONS}")

    value = macs["value"]
    source = macs["source"]

    mult_factor = FLOP_CONVENTIONS[convention]
    total_mac_count = value
    total_flops = total_mac_count * mult_factor


    if "per_layer" in macs:
        per_layer = {}
        for layer in macs["per_layer"]:
            per_layer[layer] = macs["per_layer"][layer] * mult_factor

    return {
        "value": total_flops,
        "source": source,
        "status": "computed",
        "convention": convention,
        "flops_per_mac": mult_factor,
        "per_layer": per_layer,
        "note": "a count of operations contains no unit of time"
    }



        
