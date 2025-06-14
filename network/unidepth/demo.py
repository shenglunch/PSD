import os
import json

import numpy as np
import torch
from PIL import Image


from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole


def demo(model):
    rgb = np.array(Image.open("/public/home/csl/paper-robust/DC-rebuttal5/network/unidepth/assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics_torch = torch.from_numpy(np.load("/public/home/csl/paper-robust/DC-rebuttal5/network/unidepth/assets/demo/intrinsics.npy"))
    print('intrinsics_torch', intrinsics_torch.shape)
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
    
    # infer method of V1 uses still the K matrix as input
    if isinstance(model, (UniDepthV2old, UniDepthV1)):
        camera = camera.K.squeeze(0)

    # predict
    predictions = model.infer(rgb_torch, camera)

    # get GT and pred
    depth_pred = predictions["depth"].squeeze().cpu().numpy()
    depth_gt = np.array(Image.open("/public/home/csl/paper-robust/DC-rebuttal5/network/unidepth/assets/demo/depth.png")).astype(float) / 1000.0
    print('depth_pred', depth_pred.shape)

    dfeat = predictions["psd_feat"]
    for d in dfeat:
        print(d.shape)

    # compute error, you have zero divison where depth_gt == 0.0
    depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
    depth_arel[depth_gt == 0.0] = 0.0

    # colorize
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
    depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

    # save image with pred and error
    artifact = image_grid([rgb, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
    Image.fromarray(artifact).save("/public/home/csl/paper-robust/DC-rebuttal5/network/unidepth/assets/demo/output.png")

    print("Available predictions:", list(predictions.keys()))
    print(f"ARel: {depth_arel[depth_gt > 0].mean() * 100:.2f}%")


if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    # type_ = "l"  # available types: s, b, l
    # name = f"unidepth-v2-vit{type_}14"
    # model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    with open(os.path.join('/public/home/csl/paper-robust/DC-rebuttal5/network/unidepth/configs/config_v2_vitl14.json')) as f:
        config = json.load(f)
    model = UniDepthV2(config)
    info = model.load_state_dict(torch.load('/public/home/csl/pretrained/unidepth_v2_vitl14.bin'), strict=False)
    print(f"UniDepthV2 is loaded with:")
    print(f"\t missing keys: {info.missing_keys}")
    print(f"\t additional keys: {info.unexpected_keys}")

    # set resolution level (only V2)
    # model.resolution_level = 9

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    demo(model)
