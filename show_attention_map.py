import os
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import Adam
import editdistance
import numpy as np
import cv2

from hw_line_modules import HWLine_Module, init_net
from hw_line_dataset import BaseDataset, letters

CHANNEL_SIZE = 4
HIDDEN_SIZE = 256
OUTPUT_SIZE = len(letters) + 3  # +3 for the special tokens
MAX_LENGTH = 912

BATCH_SIZE = 56
EPOCH = 240
LR = 0.0001
BETA1 = 0.5
CKPT = 'ckpt/hw_line_ep_95.pt'

train_image_path = '/home/SSD/yutao/IAM-lines/crop/images/train'
train_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-train.txt'
test_image_path = '/home/SSD/yutao/IAM-lines/crop/images/test'
test_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-test.txt'
pretrain_encoder = 'files/epoch-220.pt'

def main():
    vae = AutoencoderKL.from_pretrained("./diffusion", subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to("cuda")
    model = HWLine_Module(CHANNEL_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, max_length=MAX_LENGTH).to("cuda")
    model.load_state_dict(torch.load(CKPT), strict=False)

    optimizer = Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))

    val_dataset = BaseDataset(test_image_path, test_label_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=val_dataset.collate_fn_, num_workers=8, pin_memory=True)

    # validate
    model.eval()
    tdecs = []
    transcrs = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images = data["img"].to("cuda")
            lexicon, lexicon_length = data["lexicon"].to("cuda"), data["lexicon_length"].to("cuda")
            transcr = data["transcr"]

            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

            texts = torch.full(lexicon.shape, 2).to("cuda")
            texts[:, 0] = torch.LongTensor(lexicon.shape[0]).fill_(val_dataset.dict['[GO]'])

            pred_forradical, attention_maps = model.predict(latents, texts)
            save_dir = f'images/{transcr[0]}'
            os.makedirs(save_dir, exist_ok=True)
            for i, length in enumerate(lexicon_length.data): # 第i个样本的每个time step的attention map
                att_maps = attention_maps[0:length] 
                word_idx = data["word_idx"][i]
                for j, (h_idx, t_idx) in enumerate(word_idx):
                    alpha_map = torch.zeros_like(att_maps[0][i])
                    for k in range(h_idx, t_idx+1):
                        alpha_map += att_maps[k][i]
                    new_map_t = model.decoder_forradical.cal(images, alpha_map)
                    image = (images[0] / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
                    image = (image * 255).astype(np.uint8)
                    new_map_t = new_map_t.permute(1, 2, 0).detach().cpu().numpy()
                    visualize_attention_map(image, new_map_t, transcr[i][h_idx:t_idx+1], save_dir)
                    glyph_word = data['glyph_words'][j]
                    glyph_word = (glyph_word / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
                    glyph_word = (glyph_word * 255).astype(np.uint8)
                    cv2.imwrite(f'{save_dir}/glyph_{transcr[i][h_idx:t_idx+1]}.png', glyph_word)

                # for j, alpha_maps in enumerate(att_maps): # 第i个样本的第j个time step的attention map
                #     alpha_map = alpha_maps[i]   
                #     new_map_t = model.decoder_forradical.cal(images, alpha_map)
                #     image = (images[0] / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
                #     image = (image * 255).astype(np.uint8)
                #     new_map_t = new_map_t.permute(1, 2, 0).detach().cpu().numpy()
                #     visualize_attention_map(image, new_map_t, transcr[i][j])
            texts = val_dataset.decode(pred_forradical)
            tdecs += list(texts)
            transcrs += list(transcr)
        
    # Calculate CER and WER
    cer, wer = [], []
    cntc, cntw = 0, 0
    for text, t in zip(tdecs, transcrs):
        cc = float(editdistance.eval(text, t))
        ww = float(editdistance.eval(text.split(' '), t.split(' ')))
        cntc += len(t)
        cntw +=  len(t.split(' '))
        cer += [cc]
        wer += [ww]

    cer = sum(cer) / cntc
    wer = sum(wer) / cntw
    print('CER: %f', cer)
    print('WER: %f', wer)


def visualize_attention_map(image, attention_map, str, save_dir):
    normed_mask = np.uint8(attention_map)
    normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    normed_mask = cv2.addWeighted(image, 0.6, normed_mask, 1.0, 0)
    cv2.imwrite(f'{save_dir}/attentional_image_str_{str}.png', normed_mask)

if __name__ == '__main__':
    main()