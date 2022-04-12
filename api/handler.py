import math

import numpy as np
import torch
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

model_path = './model_file/RealESRGAN_x4plus.pth'
tile = 8
tile_pad = 10
pre_pad = 10

class ESRGANHandler:
    def __init__(self, model_path=model_path, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.scale = 4
        self.half = half
        
        state_dict = torch.load(model_path)
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.model.load_state_dict(state_dict['params_ema'])
        self.model.eval()
        if self.half:
            self.model = self.model.half()
    
    def preprocessing(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
            
        # pre-pad
        self.img = torch.nn.functional.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        
        if self.half:
            self.img = self.img.to(torch.float16)
            
    def inference(self):
        # model_inference
        self.output = self.model(self.img)
        
    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_height, output_width = height * self.scale, width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        
        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)
        
        for x in range(tiles_x):
            for y in range(tiles_y):
                ofs_x = x * self.tile
                ofs_y = y * self.tile
                
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile, height)
                
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)
                
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')
                
     
                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output
    
    @torch.no_grad()
    def enhance(self, img):
        # img: numpy
        h_input, w_input = img.shape[0:2]
        img = img.astype(np.float32)
        if np.max(img) > 256: # 16-bit
            max_range = 65535
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2: # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4: # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.preprocessing(img)
        self.tile_process()
        
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            
        if img_mode == 'RGBA':
            self.preprocessing(alpha)
            self.tile_process()
            output_alpha = self.post_process()
            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha
        
        if max_range == 65535: # 16-bit
            output = (output_img * 65535.0).round().astype(np.uint16)
        
        else:
            output = (output_img * 255.0).round().astype(np.uint8)
        
        return output, img_mode

if __name__ == '__main__':
    path = 'image/test.png'
    save_path = 'image/resolution.png'
    upsampler = ESRGANHandler(half=True)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img)
    cv2.imwrite(save_path, output)
    print('Done!')