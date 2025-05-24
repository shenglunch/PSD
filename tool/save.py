from PIL import Image
import PIL.Image as pil
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib 
import torch
import numpy as np
import torchvision
matplotlib.use('pdf')

# read image
# file='/user42/slchen/data/KITTI/KITTI/completion/data_depth_selection/val_selection_cropped/groundtruth_depth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000791_image_03.png'
# file = '/user42/slchen/csl/15refine22/pred.png'
# img_file = Image.open(file)
# depth_png = np.array(img_file, dtype=int)
# img_file.close()
# depth = depth_png.astype(np.float) / 256.
# depth = np.expand_dims(depth, -1)
# depth = torch.from_numpy(depth.copy())
# depth = depth.squeeze(-1).unsqueeze(0)
# b, h, w = depth.shape

map = np.array(
        [[0,0,0,114],
         [0,0,1,185],
         [1,0,0,114],
         [1,0,1,174],
         [0,1,0,114],
         [0,1,1,185],
         [1,1,0,114],
         [1,1,1,0]], dtype=np.float32)
m, n = map.shape
map_last = np.array([i[-1] for i in map])
sum = map_last.sum()

weights = np.zeros(m)
weights[:m-1] = sum / map_last[:m-1]
cumsum = np.zeros(m)
for i in range(m-1):
    cumsum[i+1] = map_last[:i+1].sum() / sum

def save_depth(depth, file):
    '''
    depth: [b, 1, h, w]
    '''
    depth = depth.squeeze(1)
    b, h, w = depth.shape

    # norm
    max_val = depth.max()
    depth = depth / max_val
    depth = torch.clamp(depth, min=0)
    depth = torch.clamp(depth, max=1.0)
    depth = depth.detach().numpy()

    color_image = np.zeros([b, h, w, 3], dtype=np.float32)
    for i in range(m-1):
        mask = np.logical_and(depth >= cumsum[i], depth < cumsum[i+1])
        if mask.sum() > 0:
            val = depth[mask]
            w = 1.0 - (val-cumsum[i])*weights[i]
            r = w*map[i][0]+(1.0-w)*map[i+1][0] 
            g = w*map[i][1]+(1.0-w)*map[i+1][1] 
            b = w*map[i][2]+(1.0-w)*map[i+1][2] 
            
            color_image[mask] = np.stack((r,g,b), axis=-1)

    color_image[depth <= 0] = 255.
    # color_image = color_image.squeeze(0)    
    color_image = torch.from_numpy(np.ascontiguousarray(color_image)).permute(0, 3, 1, 2)

    # plt.imshow(color_image) 
    # plt.axis('off')
    # plt.savefig("color_image.png", bbox_inches = 'tight', pad_inches = 0, dpi=192)
    torchvision.utils.save_image(color_image, file)

    return color_image

def save_depth_plt(depth, file, cmap):
    mask = depth.cpu().squeeze().numpy()>0

    plt.imshow(depth.cpu().squeeze(), cmap=cmap)
    plt.axis('off')
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8) # argb
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    rgb_image = rgb_image * mask[:,:,None]
    im = Image.fromarray(rgb_image)
    im.save(file)

def save_rgb(rgb, file):
    rgb = rgb.cpu().squeeze(0).permute(1, 2, 0).numpy()*255
    im = Image.fromarray(np.uint8(rgb))
    im.save(file)

def save_depth_kitti(depth, file):
    depth = depth.cpu().squeeze().numpy()
    depth = (depth * 256.0).astype('uint16')
    depth_buffer = depth.tobytes()
    depthsave = Image.new("I", depth.T.shape)
    depthsave.frombytes(depth_buffer, 'raw', "I;16")
    depthsave.save(file)