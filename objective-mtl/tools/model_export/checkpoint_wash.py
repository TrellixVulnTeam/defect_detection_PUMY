from collections import OrderedDict
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="input model")
    parser.add_argument("--save_path", required=True, help="output model")
    group_prefix = parser.add_mutually_exclusive_group()
    group_prefix.add_argument(
        "--prefix", default=None, help="remove the prefix from the keys"
    )
    group_prefix.add_argument(
        "--keep_prefix_list",
        default=None,
        nargs="+",
        help="keep the params with the prefix and remove the prefix from the keys",
    )
    parser.add_argument(
        "--keep_prefix", action="store_true", help="whether keep the prefix for keys"
    )
    args = parser.parse_args()
    return args


def wash_checkpoint_prefix(snapshot, save_snapshot, prefix="backbone."):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()

    if "state_dict" in checkpoint:
        print("=> state_dict")
        state_dict = checkpoint["state_dict"]
    else:
        print("=> raw")
        state_dict = checkpoint

    for k, v in state_dict.items():
        print(k)
        if k.startswith(prefix):
            name = k[len(prefix) :]
        else:
            name = k
        new_state_dict[name] = v

    torch.save({"state_dict": new_state_dict}, save_snapshot)


def wash_checkpoint_keep_prefix(
    snapshot, save_snapshot, keep_prefix_list=None, keep_prefix=False
):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    state_dict = checkpoint["state_dict"]
    print("Keeped params with prefix:", keep_prefix_list)

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        for keep_prefix_one in keep_prefix_list:
            if k.startswith(keep_prefix_one):
                # remove prefix
                if keep_prefix:
                    new_state_dict[k] = state_dict[k]
                else:
                    new_state_dict[k[len(keep_prefix_one) :]] = state_dict[k]
                break

    torch.save({"state_dict": new_state_dict}, save_snapshot)


if __name__ == "__main__":
    args = get_args()
    if args.keep_prefix_list is not None:
        wash_checkpoint_keep_prefix(
            args.checkpoint, args.save_path, args.keep_prefix_list, args.keep_prefix
        )
    else:
        wash_checkpoint_prefix(args.checkpoint, args.save_path, args.prefix)
