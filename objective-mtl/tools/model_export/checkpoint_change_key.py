import sys
import torch


def change_checkpoint_key(
    snapshot, save_snapshot, ori_key="", to_key="state_dict"
):
    checkpoint = torch.load(snapshot, map_location=torch.device("cpu"))
    if ori_key == "":
        state_dict = checkpoint
    else:
        state_dict = checkpoint[ori_key]
    torch.save({to_key: state_dict}, save_snapshot)


if __name__ == "__main__":
    change_checkpoint_key(sys.argv[1], save_snapshot=sys.argv[2])
