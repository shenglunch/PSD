import tyro
import os
import glob
from tqdm.auto import tqdm

from promptda.utils.io_wrapper import load_image, load_depth, save_depth
from promptda.utils.parallel_utils import parallel_execution
from promptda.promptda import PromptDA

def load_data(input_path: str, max_size: int):
    root_dir = os.path.dirname(input_path)
    scene_name = input_path.split('/')[-1].split('.')[0]
    input_dir = os.path.join(root_dir, scene_name)
    if not os.path.exists(input_dir):
        cmd = f'unzip -o {input_path} -d {root_dir}'
        os.system(cmd)

    if not os.path.exists(os.path.join(input_dir, 'rgb')):
        os.makedirs(os.path.join(input_dir, 'rgb'), exist_ok=True)
        cmd = f'ffmpeg -i {input_dir}/rgb.mp4 -start_number 0 -q:v 2 {input_dir}/rgb/%06d.jpg'
        os.system(cmd)

    rgb_files = sorted(glob.glob(os.path.join(input_dir, 'rgb', '*.jpg')))
    prompt_depth_files = sorted(glob.glob(os.path.join(input_dir, 'depth', '*.png')))

    if len(rgb_files) != len(prompt_depth_files):
        min_len = min(len(rgb_files), len(prompt_depth_files))
        rgb_files = rgb_files[:min_len]
        prompt_depth_files = prompt_depth_files[:min_len]

    rgbs = parallel_execution(rgb_files, 
                              to_tensor=True, # to_tensor
                              max_size=max_size, 
                              action=load_image, 
                              num_processes=32, 
                              print_progress=True,
                              desc='Loading RGB images')

    prompt_depths = parallel_execution(prompt_depth_files, 
                              to_tensor=True, # to_tensor
                              action=load_depth, 
                              num_processes=32, 
                              print_progress=True,
                              desc='Loading Prompt Depth')
    return rgbs, prompt_depths


def main(input_path: str = 'data/8b98276b0a.zip', 
         output_path: str = 'data/8b98276b0a_results',
         max_size: int = 1008,
         ):
    os.makedirs(output_path, exist_ok=True)
    rgbs, prompt_depths = load_data(input_path, max_size)
    results = []
    DEVICE = 'cuda'
    model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()
    for frame_idx, (rgb, prompt_depth) in tqdm(enumerate(zip(rgbs, prompt_depths)), desc='Inferring', total=len(rgbs)):
        rgb, prompt_depth = rgb.to(DEVICE), prompt_depth.to(DEVICE)
        depth = model.predict(rgb, prompt_depth)
        save_depth(depth.detach().cpu(), 
                   output_path=os.path.join(output_path, f'{frame_idx:06d}.png'),
                   save_vis=True)

if __name__ == "__main__":
    tyro.cli(main)