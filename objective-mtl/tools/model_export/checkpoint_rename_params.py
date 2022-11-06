import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_checkpoint", required=True, help="input model")
    parser.add_argument("--output_checkpoint", required=True, help="output model")
    parser.add_argument(
        "--change_key_name", action="store_true", help="changing prefix or params"
    )

    parser.add_argument(
        "--from_prefix_list", default=None, nargs="+", help="original prefix keys"
    )
    parser.add_argument(
        "--to_prefix_list", default=None, nargs="+", help="to prefix keys"
    )
    args = parser.parse_args()
    return args


def rename_checkpoint_params_prefix(
    snapshot, save_snapshot, prefix_origin=None, to_prefix=None
):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        is_key = False
        for i in range(len(prefix_origin)):
            if key.startswith(prefix_origin[i]):
                new_state_dict[to_prefix[i] + key[len(prefix_origin[i]) :]] = value
                is_key = True
                break
        if not is_key:
            new_state_dict[key] = value

    torch.save({"state_dict": new_state_dict}, save_snapshot)


def rename_checkpoint_params(
    snapshot, save_snapshot, param_key_list=None, to_key_llist=None
):

    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        matched = False
        for i, param in enumerate(param_key_list):
            if key == param:
                new_state_dict[to_key_llist[i]] = value
                matched = True
                break

        if not matched:
            new_state_dict[key] = value

    torch.save({"state_dict": new_state_dict}, save_snapshot)


if __name__ == "__main__":
    # input_checkpoint = 'meta/models/swin_large_patch4_window7_224_22kto1k.pth'
    # output_checkpoint = 'meta/models/swin_large_patch4_window7_224_imagenet1k.pth'
    # rename_checkpoint_params(
    #     input_checkpoint,
    #     save_snapshot=output_checkpoint,
    #     param_key_list=['norm.weight', 'norm.bias'],
    #     to_key_llist=['norm3.weight', 'norm3.bias'])

    # input_checkpoint = "meta/pretrained/checkpoint_0049_00530150.pth.tar"
    # output_checkpoint = "meta/pretrained/swin_t_moco_epoch_50.pth"

    args = get_args()

    if args.change_key_name:
        if args.from_prefix_list is None:
            keys_origin = ["norm.weight", "norm.bias"]
        else:
            keys_origin = args.from_prefix_list
        if args.to_prefix_list is None:
            to_keys = ["norm3.weight", "norm3.bias"]
        else:
            to_keys = args.to_prefix_list

        rename_checkpoint_params(
            args.input_checkpoint,
            save_snapshot=args.output_checkpoint,
            param_key_list=keys_origin,
            to_key_llist=to_keys,
        )
    else:
        if args.from_prefix_list is None:
            prefix_origin = ["stages.", "patch_embed.projection."]
        else:
            prefix_origin = args.from_prefix_list
        if args.to_prefix_list is None:
            to_prefix = ["layers.", "patch_embed.proj."]
        else:
            to_prefix = args.to_prefix_list

        rename_checkpoint_params_prefix(
            args.input_checkpoint,
            save_snapshot=args.output_checkpoint,
            prefix_origin=prefix_origin,
            to_prefix=to_prefix,
        )
