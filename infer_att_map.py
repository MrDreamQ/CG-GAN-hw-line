import os 
import glob
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import Adam
import editdistance
from PIL import Image
import numpy as np
import cv2
import random

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
IMG_DIR = '/home/SSD/yutao/workspace/One-DM-line/Generated/train-20240730_030728/24/hwd'
imgs = glob.glob(os.path.join(IMG_DIR, '*.png'))
random.shuffle(imgs)

train_image_path = '/home/SSD/yutao/IAM-lines/crop/images/train'
train_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-train.txt'
test_image_path = '/home/SSD/yutao/IAM-lines/crop/images/test'
test_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-test.txt'


def main():
    vae = AutoencoderKL.from_pretrained("./diffusion", subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to("cuda")
    model = HWLine_Module(CHANNEL_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, max_length=MAX_LENGTH).to("cuda")
    model.load_state_dict(torch.load(CKPT), strict=False)

    val_dataset = BaseDataset(test_image_path, test_label_path)

    def visualize_glyph(transcr, img_dir):
        content = []
        for char in transcr:
            idx = val_dataset.letter2index[char]
            con_symbol = val_dataset.con_symbols[idx].numpy()
            thr = con_symbol==1.0
            prof = thr.sum(axis=0)
            on = np.argwhere(prof)[:,0]
            if len(on)>0:
                left = np.min(on)
                right = np.max(on)
                con_symbol = con_symbol[:,left-2:right+2]
            if len(on) == 0:
                con_symbol = con_symbol[:, 2:14]
            con_symbol = torch.from_numpy(con_symbol)
            content.append(con_symbol)
        content = torch.cat(content, dim=-1)
        content = content.numpy()
        ratio_w = content.shape[1]*4
        if ratio_w <= 912:
            content = cv2.resize(content, (ratio_w, 64))
        else:
            content = cv2.resize(content, (912, 64))
        content = 1. - content
        content = np.stack((content, content, content), axis=2)
        content = (content*255).astype(np.uint8)
        cv2.imwrite(os.path.join(img_dir, f'glyph_{transcr}.png'), content)

    # validate
    model.eval()
    with torch.no_grad():
        for img_path in imgs:
            img_path = str(img_path)
            img_name = os.path.basename(img_path)
            transcr = img_name.split('_')[0].split('-',1)[-1]
            # img_path = '/home/SSD/yutao/IAM-lines/crop/images/train/000/a01-014u-08.png'
            # transcr = 'at Chequers.'
            words = transcr.split(' ')
            h_ids, t_ids = [], []
            for word in words:
                h_str = word[0]
                t_str = word[-1]
                h_idx = transcr.index(h_str, t_ids[-1] if t_ids else 0)
                t_idx = h_idx + len(word) - 1
                h_idx = 0
                h_ids.append(h_idx)
                t_ids.append(t_idx)
            word_idx = [(h, t) for h, t in zip(h_ids, t_ids)]
            image = Image.open(img_path).convert('RGB')
            images = val_dataset.transforms(image).unsqueeze(0).to("cuda")
            tmp = torch.ones((1, 3, 64, 912)).to("cuda")
            tmp[:, :, :, 0:images.shape[-1]] = images
            images = tmp
            text = [transcr]
            lexicon, lexicon_length = val_dataset.encode(text)
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            texts = torch.full([lexicon.shape[0], 90], 2).to("cuda")
            texts[:, 0] = torch.LongTensor(lexicon.shape[0]).fill_(val_dataset.dict['[GO]'])
            pred_forradical, attention_maps = model.predict(latents, texts)

            # for i in range(pred_forradical.shape[0]):
            #     space_pos = torch.where(pred_forradical[i] == val_dataset.dict[' '])[0].tolist()
            #     end_pos = torch.where(pred_forradical[i] == val_dataset.dict['[END]'])[0].tolist()
            #     end_pos = [end_pos[0]] if len(end_pos) > 0 else [pred_forradical.shape[1]]
            #     space_pos = [pos-1 for pos in space_pos]
            #     end_pos = [pos-1 for pos in end_pos]
            #     space_end_pos = space_pos + end_pos
            #     att_map_ht = (0,114)
            #     cover_words = ''
            #     if len(space_end_pos) <= 2:
            #         att_map_ht = (0, space_end_pos[-1])
            #         cover_idx = word_idx[-1]
            #         cover_words = transcr[0:cover_idx[1]+1]
            #     else:
            #         random_space_idx = random.randint(min(2, len(space_end_pos)//2), max(2, len(space_end_pos)//2))
            #         att_map_ht = (0, space_end_pos[random_space_idx])
            #         cover_idx = word_idx[random_space_idx]
            #         cover_words = transcr[0:cover_idx[1]+1]
            #     att_maps = attention_maps[att_map_ht[0]:att_map_ht[1]+1]
            #     alpha_map = torch.zeros_like(att_maps[0][i])
            #     for k in range(att_map_ht[0], att_map_ht[1]+1):
            #         alpha_map += att_maps[k][i]
            #     new_map_t = model.decoder_forradical.cal(images, alpha_map)
            #     new_map_t = new_map_t.permute(1, 2, 0).detach().cpu().numpy()
            #     image = (images[0] / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
            #     image = (image * 255).astype(np.uint8)
            #     visualize_attention_map(image, new_map_t, cover_words, f'images/{transcr}/')
            #     w_ratio = len(cover_words) / len(transcr)
            #     glyph = val_dataset.get_glyph(cover_words, w_ratio)
            #     glyph = glyph.permute(1, 2, 0).detach().cpu().numpy()
            #     glyph = ((glyph/2+0.5)*255).astype(np.uint8)
            #     cv2.imwrite(os.path.join(f'images/{transcr}/', f'glyph_{cover_words}.png'), glyph)
                
            for i, length in enumerate(lexicon_length.data): # 第i个样本的每个time step的attention map
                att_maps = attention_maps[0:length] 
                att_map_ht = (0,114)
                cover_words = ''
                if len(word_idx) <= 2:
                    cover_idx = word_idx[-1]
                    att_map_ht = (cover_idx[0], cover_idx[1])
                    cover_words = transcr[cover_idx[0]:cover_idx[1]+1]
                else:
                    random_idx = random.randint(min(2, len(word_idx)//2), max(2, len(word_idx)//2))
                    cover_idx = word_idx[random_idx]
                    att_map_ht = (cover_idx[0], cover_idx[1])
                    cover_words = transcr[cover_idx[0]:cover_idx[1]+1]
                alpha_map = torch.zeros_like(att_maps[0][i])
                for k in range(att_map_ht[0], att_map_ht[1]+1):
                    alpha_map += att_maps[k][i]
                new_map_t = model.decoder_forradical.cal(images, alpha_map)
                new_map_t = new_map_t.permute(1, 2, 0).detach().cpu().numpy()
                image = (images[0] / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255).astype(np.uint8)
                visualize_attention_map(image, new_map_t, cover_words, f'images/{transcr}/')
                w_ratio = len(cover_words) / len(transcr)
                glyph = val_dataset.get_glyph(cover_words, w_ratio)
                glyph = glyph.permute(1, 2, 0).detach().cpu().numpy()
                glyph = ((glyph/2+0.5)*255).astype(np.uint8)
                cv2.imwrite(os.path.join(f'images/{transcr}/', f'glyph_{cover_words}.png'), glyph)

                # for j, (h_idx, t_idx) in enumerate(word_idx):
                #     alpha_map = torch.zeros_like(att_maps[0][i])
                #     for k in range(h_idx, t_idx+1):
                #         alpha_map += att_maps[k][i]
                #     new_map_t = model.decoder_forradical.cal(images, alpha_map)
                #     image = (images[0] / 2 + 0.5).clamp(0,1).permute(1, 2, 0).detach().cpu().numpy()
                #     image = (image * 255).astype(np.uint8)
                #     new_map_t = new_map_t.permute(1, 2, 0).detach().cpu().numpy()
                #     visualize_attention_map(image, new_map_t, transcr[h_idx:t_idx+1], f'images/{transcr}/')
                #     w_ratio = len(cover_words) / len(transcr)
                #     glyph = val_dataset.get_glyph(cover_words, w_ratio)
                #     glyph = glyph.permute(1, 2, 0).detach().cpu().numpy()
                #     glyph = ((glyph/2+0.5)*255).astype(np.uint8)
                #     cv2.imwrite(os.path.join(f'images/{transcr}/', f'glyph_{cover_words}.png'), glyph)

            texts = val_dataset.decode(pred_forradical)
            print(texts)
        

def visualize_attention_map(image, attention_map, str, img_dir):
    normed_mask = np.uint8(attention_map)
    normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    normed_mask = cv2.addWeighted(image, 0.6, normed_mask, 1.0, 0)
    os.makedirs(img_dir, exist_ok=True)
    cv2.imwrite(f'{img_dir}/attentional_image_str_{str}.png', normed_mask)


if __name__ == '__main__':
    main()