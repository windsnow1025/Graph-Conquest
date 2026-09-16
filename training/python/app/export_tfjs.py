"""Export PyTorch model weights to TF.js layers-model format (v9, context-free trunk)."""
import json
import os
import numpy as np
import torch

from app.config import (
    CONTEXT_OFFSET, NUM_ACTION_TYPES,
    MOVE_TARGET_DIM, BATTLE_TARGET_DIM,
    HIDDEN1, HIDDEN2, HEAD_HIDDEN,
    CTX_DT_LEN, CTX_REC_LEN, CTX_ARMY_LEN, CTX_MOV_LEN,
    CTX_BTGT_LEN, CTX_BSEL_LEN, CTX_BALLOC_LEN, CTX_BRET_LEN,
)


def _dense_config(name, units, activation, input_name):
    return {
        "name": name,
        "class_name": "Dense",
        "config": {
            "units": units, "activation": activation, "use_bias": True,
            "kernel_initializer": {"class_name": "VarianceScaling",
                "config": {"scale": 1, "mode": "fan_avg", "distribution": "normal", "seed": None}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None, "kernel_constraint": None, "bias_constraint": None,
            "name": name, "trainable": True,
        },
        "inbound_nodes": [[[input_name, 0, 0, {}]]],
    }


def _concat_config(name, input_names):
    return {
        "name": name,
        "class_name": "Concatenate",
        "config": {"axis": -1, "name": name},
        "inbound_nodes": [[[n, 0, 0, {}] for n in input_names]],
    }


def _input_config(name, size):
    return {
        "name": name,
        "class_name": "InputLayer",
        "config": {"batch_input_shape": [None, size], "dtype": "float32", "sparse": False, "name": name},
        "inbound_nodes": [],
    }


def build_model_json():
    layers = [
        _input_config("state_input", CONTEXT_OFFSET),
        _input_config("ctx_dt", CTX_DT_LEN),
        _input_config("ctx_rec", CTX_REC_LEN),
        _input_config("ctx_army", CTX_ARMY_LEN),
        _input_config("ctx_mov", CTX_MOV_LEN),
        _input_config("ctx_btgt", CTX_BTGT_LEN),
        _input_config("ctx_bsel", CTX_BSEL_LEN),
        _input_config("ctx_balloc", CTX_BALLOC_LEN),
        _input_config("ctx_bret", CTX_BRET_LEN),
    ]

    # Trunk
    layers.append(_dense_config("dense1", HIDDEN1, "relu", "state_input"))
    layers.append(_dense_config("dense2", HIDDEN2, "relu", "dense1"))

    # Per-head: (head_name, head_units, head_activation, ctx_input_names)
    head_configs = [
        ("value",            1,                "sigmoid", []),
        ("action_type",      NUM_ACTION_TYPES, "linear",  ["ctx_dt", "ctx_army"]),
        ("split_fraction",   1,                "sigmoid", ["ctx_dt", "ctx_army"]),
        ("disband_fraction", 1,                "sigmoid", ["ctx_dt", "ctx_army"]),
        ("recruit_fraction", 1,                 "sigmoid", ["ctx_dt", "ctx_rec"]),
        ("move_target",      MOVE_TARGET_DIM,   "linear",  ["ctx_dt", "ctx_mov"]),
        ("battle_target",    BATTLE_TARGET_DIM, "linear",  ["ctx_dt", "ctx_btgt"]),
        ("battle_select",    1,                 "sigmoid", ["ctx_dt", "ctx_bsel"]),
        ("kill_fraction",    1,                "sigmoid", ["ctx_dt", "ctx_balloc"]),
        ("battle_retreat",   1,                "sigmoid", ["ctx_dt", "ctx_bret"]),
    ]

    for name, units, act, ctx_names in head_configs:
        if ctx_names:
            concat_name = f"{name}_concat"
            layers.append(_concat_config(concat_name, ["dense2"] + ctx_names))
            layers.append(_dense_config(f"{name}_hidden", HEAD_HIDDEN, "relu", concat_name))
        else:
            layers.append(_dense_config(f"{name}_hidden", HEAD_HIDDEN, "relu", "dense2"))
        layers.append(_dense_config(f"{name}_head", units, act, f"{name}_hidden"))

    input_names = ["state_input", "ctx_dt", "ctx_rec", "ctx_army", "ctx_mov",
                   "ctx_btgt", "ctx_bsel", "ctx_balloc", "ctx_bret"]
    output_names = [f"{cfg[0]}_head" for cfg in head_configs]

    return {
        "class_name": "Model",
        "config": {
            "name": "graph_conquest_nn_v9",
            "layers": layers,
            "input_layers": [[n, 0, 0] for n in input_names],
            "output_layers": [[n, 0, 0] for n in output_names],
        },
        "keras_version": "tfjs-layers 4.22.0",
        "backend": "tensor_flow.js",
    }


WEIGHT_ORDER = [
    "dense1", "dense2",
    "value_hidden", "value_head",
    "action_type_hidden", "action_type_head",
    "split_fraction_hidden", "split_fraction_head",
    "disband_fraction_hidden", "disband_fraction_head",
    "recruit_fraction_hidden", "recruit_fraction_head",
    "move_target_hidden", "move_target_head",
    "battle_target_hidden", "battle_target_head",
    "battle_select_hidden", "battle_select_head",
    "kill_fraction_hidden", "kill_fraction_head",
    "battle_retreat_hidden", "battle_retreat_head",
]


def export_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state_dict = model.state_dict()
    weight_specs = []
    weight_buffers = []

    for layer_name in WEIGHT_ORDER:
        kernel = state_dict[f"{layer_name}.weight"].cpu().numpy().T
        bias = state_dict[f"{layer_name}.bias"].cpu().numpy()
        weight_specs.append({"name": f"{layer_name}/kernel", "shape": list(kernel.shape), "dtype": "float32"})
        weight_buffers.append(kernel.astype(np.float32).tobytes())
        weight_specs.append({"name": f"{layer_name}/bias", "shape": list(bias.shape), "dtype": "float32"})
        weight_buffers.append(bias.astype(np.float32).tobytes())

    with open(os.path.join(output_dir, "weights.bin"), "wb") as f:
        for buf in weight_buffers:
            f.write(buf)

    model_json = {
        "modelTopology": build_model_json(),
        "format": "layers-model",
        "generatedBy": "TensorFlow.js tfjs-layers v4.22.0",
        "convertedBy": None,
        "weightsManifest": [{"paths": ["weights.bin"], "weights": weight_specs}],
    }
    with open(os.path.join(output_dir, "model.json"), "w") as f:
        json.dump(model_json, f, indent=2)

    total_bytes = sum(len(b) for b in weight_buffers)
    print(f"Exported to {output_dir}: model.json + weights.bin ({total_bytes} bytes)")


def import_tfjs_weights(model, model_dir):
    with open(os.path.join(model_dir, "model.json"), "r") as f:
        mj = json.load(f)
    with open(os.path.join(model_dir, "weights.bin"), "rb") as f:
        raw = f.read()

    offset = 0
    state_dict = {}
    for spec in mj["weightsManifest"][0]["weights"]:
        name = spec["name"]
        shape = spec["shape"]
        n_floats = 1
        for s in shape:
            n_floats *= s
        n_bytes = n_floats * 4
        arr = np.frombuffer(raw[offset:offset + n_bytes], dtype=np.float32).reshape(shape)
        offset += n_bytes
        if name.endswith("/kernel"):
            layer = name.replace("/kernel", "")
            state_dict[f"{layer}.weight"] = torch.from_numpy(arr.T.copy())
        elif name.endswith("/bias"):
            layer = name.replace("/bias", "")
            state_dict[f"{layer}.bias"] = torch.from_numpy(arr.copy())

    model.load_state_dict(state_dict)
    print(f"Loaded TF.js weights from {model_dir}")
