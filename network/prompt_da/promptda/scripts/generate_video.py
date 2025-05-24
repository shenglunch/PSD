import tyro
from tqdm.auto import tqdm
import numpy as np
import glob
import os
import cv2
import imageio
from promptda.utils.depth_utils import smooth_min_max, visualize_depth
from promptda.utils.io_wrapper import load_depth, load_image
from promptda.utils.parallel_utils import async_call, parallel_execution


def load_depths(depth_paths: list[str]) -> list[np.ndarray]:
    depths = parallel_execution(depth_paths,
                                to_tensor=False,
                                action=load_depth,
                                num_processes=32,
                                desc='Loading depths')
    return depths


def load_imgs(rgb_paths: list[str], max_size: int) -> list[np.ndarray]:
    rgbs = parallel_execution(rgb_paths,
                              to_tensor=False,
                              max_size=max_size,
                              action=load_image,
                              num_processes=32,
                              desc='Loading RGB images')
    return rgbs


def load_result_depths(result_path: str) -> list[np.ndarray]:
    depth_paths = sorted(glob.glob(os.path.join(result_path, '*.png')))
    depths = load_depths(depth_paths)
    return depths


def load_prompt_depths(input_path: str) -> list[np.ndarray]:
    prompt_depth_paths = sorted(glob.glob(os.path.join(input_path, 'depth/*.png')))
    prompt_depths = load_depths(prompt_depth_paths)
    return prompt_depths


def load_rgbs(input_path: str, max_size: int) -> list[np.ndarray]:
    rgb_paths = sorted(glob.glob(os.path.join(input_path, 'rgb/*.jpg')))
    rgbs = load_imgs(rgb_paths, max_size)
    return rgbs

@async_call
def generate_frame(
    depth: np.ndarray,
    min_val: float,
    max_val: float,
    frame_idx: int,
    result_path: str,
    prompt_depth: np.ndarray = None,
    rgb: np.ndarray = None,
) -> None:
    output_img = visualize_depth(depth, min_val, max_val)
    if prompt_depth is not None:
        prompt_depth_img = visualize_depth(prompt_depth, min_val, max_val)
        if prompt_depth_img.shape[:2] != depth.shape[:2]:
            prompt_depth_img = cv2.resize(prompt_depth_img, (depth.shape[1], depth.shape[0]))
        output_img = np.concatenate([output_img, prompt_depth_img], axis=1)
    if rgb is not None:
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = (rgb * 255).astype(np.uint8)
        output_img = np.concatenate([rgb, output_img], axis=1)
    imageio.imwrite(os.path.join(result_path, f'{frame_idx:06d}_smooth.jpg'), output_img)


def process_stray_scan(input_path: str = 'data/8b98276b0a', 
                       result_path: str = 'data/8b98276b0a_results',
                       include_prompt: bool = True,
                       include_rgb: bool = True,
                       percentile: float = 2,
                       smooth_interval: int = 60) -> None:
    result_depths = load_result_depths(result_path)
    min_vals = [np.percentile(depth, percentile) for depth in result_depths]
    max_vals = [np.percentile(depth, 100 - percentile) for depth in result_depths]
    min_vals_smooth, max_vals_smooth = smooth_min_max(min_vals, max_vals, smooth_interval)

    if include_prompt:
        prompt_depths = load_prompt_depths(input_path)
    if include_rgb:
        rgbs = load_rgbs(input_path, max(result_depths[0].shape))

    min_len = min(len(result_depths), len(prompt_depths), len(rgbs))
    result_depths = result_depths[:min_len]
    prompt_depths = prompt_depths[:min_len]
    rgbs = rgbs[:min_len]

    for frame_idx in tqdm(range(len(result_depths)), desc='Generating frames'):
        generate_frame(result_depths[frame_idx],
                       min_vals_smooth[frame_idx],
                       max_vals_smooth[frame_idx],
                       frame_idx,
                       result_path,
                       prompt_depths[frame_idx] if include_prompt else None,
                       rgbs[frame_idx] if include_rgb else None)

def main() -> None:
    pass


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            'process_stray_scan': process_stray_scan,
            'main': main,
        }
    )

