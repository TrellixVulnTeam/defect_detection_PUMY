import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_checkpoint", required=True, help="input model")
    parser.add_argument("--output_checkpoint", required=True, help="output model")

    parser.add_argument(
        "--from_name_list", default=None, nargs="+", help="original name keys"
    )
    parser.add_argument("--to_name_list", default=None, nargs="+", help="to name keys")
    args = parser.parse_args()
    return args


def rename_checkpoint_params_match(
    snapshot, save_snapshot, match_origin=None, to_name=None
):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        is_key = False
        for i in range(len(match_origin)):
            if match_origin[i] in key:
                new_key = key.replace(match_origin[i], to_name[i])
                new_state_dict[new_key] = value
                is_key = True
                break
        if not is_key:
            new_state_dict[key] = value

    torch.save({"state_dict": new_state_dict}, save_snapshot)


if __name__ == "__main__":
    args = get_args()

    if args.from_name_list is None:
        match_origin = [
            "attn.w_msa.relative_position_bias_table",
            "attn.w_msa.relative_position_index",
            "attn.w_msa.qkv.weight",
            "attn.w_msa.qkv.bias",
            "attn.w_msa.proj.weight",
            "attn.w_msa.proj.bias",
            "ffn.layers.0.0.weight",
            "ffn.layers.0.0.bias",
            "ffn.layers.1.weight",
            "ffn.layers.1.bias",
        ]
    else:
        match_origin = args.from_name_list
    if args.to_name_list is None:
        to_name = [
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
    else:
        to_name = args.to_name_list

    rename_checkpoint_params_match(
        args.input_checkpoint,
        save_snapshot=args.output_checkpoint,
        match_origin=match_origin,
        to_name=to_name,
    )
