# a conversion example for https://github.com/cfzd/Ultra-Fast-Lane-Detection 
# python3 pytorch2onnx configs/tusimple.py --test_model tusimple_18.pth
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # input data size
    batch_size = 1
    input_shape = (3, 288, 800)
    # dummy input data
    x = torch.randn(batch_size, *input_shape).cuda()

    export_onnx_file = "tusimple_18.onnx"
    torch.onnx.export(
        net,
        x,
        export_onnx_file,
        verbose=False
    )
