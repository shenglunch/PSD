try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from .model.monodepth_model import get_configured_monodepth_model
from .utils.running import load_ckpt

def get_model(config, pretrained):
    net_cfg = Config.fromfile(config)
    net = get_configured_monodepth_model(net_cfg, )
    net, _,  _, _ = load_ckpt(pretrained, net, strict_match=False)
    return net