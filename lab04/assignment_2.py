from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
)

from complexity import (
    FLOPS_PER_MAC,
    FLOP_CONVENTIONS,
    BN_BUFFERS_PER_CHANNEL,
    BUFFER_DTYPE,
)

from main import (
    build_synthetic_graph
)

# typo correction: helper should be _layer_macs() NOT _layer_parameters()

def arithmetic_intensity(graph: Graph) -> dict[str, Any]:

    per_layer = {}
    convention = "mac_is_two_flops" # bc  flops_per_mac == 2 
    mult_factor = FLOP_CONVENTIONS[convention]

    whole_model_intensity = 0
    all_flops = 0 
    all_bytes = 0

    for i in range(len(graph.layers)):
        buffer_bytes = 0.0

        ly = graph.layers[i]
        
        if ly.reads:
            if ly.reads[0] == "relu1":
                input_bytes = ly.in_elements * dtype_bytes(graph.layers[2].act_dtype) # reads is tuple of strings (act_dtype fp32 strings)
        else:
            if i > 0:
                input_bytes = ly.in_elements * dtype_bytes(graph.layers[i-1].act_dtype)
            else:
                input_bytes = ly.in_elements * dtype_bytes(graph.precision)
        
        ly = graph.layers[i] # have to include twice to avoid errors for some reason

        if ly.kind == "add" and len(ly.reads) == 2:
            input_bytes = 2 * ly.in_elements * dtype_bytes(ly.act_dtype)

        if ly.kind == "bn":
            buffer_bytes = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0] * dtype_bytes(BUFFER_DTYPE)


        # MACS
        total_layer_macs = _layer_macs(ly)
        total_flops = total_layer_macs * mult_factor
        output_bytes = ly.out_elements * dtype_bytes(graph.layers[i].act_dtype)
        weight = dtype_bytes(ly.weight_dtype)
       
        bytes_moved = (input_bytes + (weight/buffer_bytes) + output_bytes) if buffer_bytes != 0 else (input_bytes + output_bytes)

        intensity = total_flops/bytes_moved

        per_layer[ly.name] = {
            "flops": total_flops,
            "bytes": bytes_moved,
            "intensity": 0.0 if bytes_moved == 0 else round(intensity, 4)
        }

        whole_model_intensity = whole_model_intensity + intensity
        all_flops = all_flops + total_flops
        all_bytes = all_bytes + bytes_moved
    
    
    value =  round(whole_model_intensity, 4)# Whole-model intensity rounded to 4 decimals
    message = f"{graph.name}: FLOPs over bytes, every byte counted once"

    returned_rest_dict = {
        "per_layer": per_layer,
        "total_flops": all_flops,
        "total_bytes": all_bytes,
        "flops_per_mac": FLOPS_PER_MAC,
        "cache_reuse_assumed": False,
        "note": "no fusion and no cache reuse; both would lower the byte column"
    }

    return computed(value, message, **returned_rest_dict)



## HELPER ##
def _layer_macs(conv: Layer) -> int:
    if conv.kind == "linear":
        macs_count = conv.in_elements * conv.out_elements # only one layer out of the eight will have kind==linear
    else:
        macs_count = 0

    return macs_count



# MAIN to test output ##
def main(output_path: str = "complexity_results.json") -> None:
    # 1. Setup graph (use synthetic, or load_graph("model.json") if available)
    graph = build_synthetic_graph()

    # Optional: validate structure first
    validation_issues = graph.validate()
    if validation_issues:
        print(f"Graph validation warnings: {validation_issues}")

    print(f"func output: {json.dumps(arithmetic_intensity(graph), indent=4)}")

if __name__ == "__main__":
    main()
        

        

            
