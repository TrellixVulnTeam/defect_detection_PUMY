from mtl.utils.reg_util import Registry, build_module_from_dict
from mtl.utils.config_util import convert_to_dict

RUNNERS = Registry("runner")


def build_runner(cfg, default_args=None):
    cfg_dict = convert_to_dict(cfg)
    return build_module_from_dict(cfg_dict, RUNNERS, default_args=default_args)
