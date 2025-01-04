'''
Codes for testing； downsampling：x4
'''

import os
import glob
import numpy as np
import cv2
import torch
from data import util
import matplotlib

matplotlib.use("Agg")

from models.modules import c1, hrnet, sft_arch
import models.modules.seg_arch as seg_arch

model_path = '/root/experiments/best_G.pth'  # Modify your root directory
test_img_folder = "/High resolution image path"  # The following is used for downsampling
save_result_path = "./test_result"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

model = sft_arch.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)
encode_model = hrnet.__dict__['hrnetv2'](pretrained=True, use_input_norm=True).cuda().eval()
decode = c1.C1().cuda().eval()
decode.load_state_dict(torch.load("root/trained_model/decoder_epoch_30.pth",  # Modify your root directory
                                  map_location=lambda storage, loc: storage), strict=False)
segnet = seg_arch.OutdoorSceneSeg().cuda()
segnet.load_state_dict(torch.load('root/trained_model/segmentation_OST_bic.pth'), strict=False)
segnet.eval()
print('Testing MYSRNET ...')

for idx, path in enumerate(glob.glob(test_img_folder + '/*')):
    imgname = os.path.basename(path)
    basename = os.path.splitext(imgname)[0]
    print(idx + 1, basename)

    img = cv2.imread(path)
    img_show = img

    # img = util.modcrop(img, 8)
    img = img * 1.0 / 255
    if img.ndim == 2:
        continue
        img = np.expand_dims(img, axis=2)

    h, w, _ = img.shape

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    # MATLAB imresize
    # You can use the MATLAB to generate LR images first for faster imresize operation
    img_LR = util.imresize(img, 1 / 4, antialiasing=True)
    img_FLR = util.imresize(img_LR, 4, antialiasing=True)

    img_FLR = img_FLR[[2, 1, 0], :, :]
    img_LR = img_LR[[2, 1, 0], :, :]
    img = img[[2, 1, 0], :, :]
    img = img.unsqueeze(0)
    img = img.to(device)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.to(device)
    img_FLR = img_FLR.unsqueeze(0)
    img_FLR = img_FLR.to(device)
    with torch.no_grad():
        fea, mfea = encode_model(img_FLR, return_feature_maps=True)
        output = model((img_LR, mfea))
        output = output.data.float().cpu().squeeze()
    output = util.tensor2img(output)
    util.save_img(output, os.path.join(save_result_path, basename + '.png'))


