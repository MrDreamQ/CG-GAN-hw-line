import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
from PIL import Image
import torchvision
import cv2

letters = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
style_len = 912


class BaseDataset(Dataset):
    def __init__(self, image_path, text_path):
        
        self.style_len = style_len
        self.letters = letters
        self.data_dict = self.load_data(text_path)
        self.image_path = os.path.join(image_path)
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        self.con_symbols = self.get_symbols()
        list_token = ['[GO]', '[END]', '[PAD]']
        self.character = list_token + list(letters)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i


    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip() for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                writer_id = i.split(' ',1)[0].split(',')[0]
                img_idx = i.split(' ',1)[0].split(',')[1]
                image = img_idx + '.png'
                transcription = i.split(' ',1)[1]
                full_dict[idx] = {'image': image, 'wid': writer_id, 'label':transcription}
                idx += 1
        return full_dict


    def get_symbols(self):
        with open(f"files/unifont.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # PAD_TOKEN image
        contents = torch.stack(contents)
        return contents


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['wid']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        contents = []
        for char in transcr:
            idx = self.letter2index[char]
            con_symbol = self.con_symbols[idx].numpy()
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
            contents.append(con_symbol)
        contents = torch.cat(contents, dim=-1)
        contents = contents.numpy()
        ratio_w = contents.shape[1]*4
        if ratio_w <= self.style_len:
            contents = cv2.resize(contents, (ratio_w, 64))
        else:
            contents = cv2.resize(contents, (self.style_len, 64))
        contents = 1. - contents
        # cv2.imwrite('glyph.png', contents*255)
        contents = np.stack((contents, contents, contents), axis=2)
        glyphs = self.transforms(contents)

        return {'img':image,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name,
                'glyph': glyphs}


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        transcr = [item['transcr'] for item in batch]
        image_name = [item['image_name'] for item in batch]
        
        imgs = torch.full([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], self.style_len], fill_value=1., dtype=torch.float32)
        
        glyphs = torch.ones([len(batch), batch[0]['glyph'].shape[0], batch[0]['glyph'].shape[1], self.style_len], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)

            glyphs[idx, :, :, 0:item['glyph'].shape[2]] = item['glyph']
            
        wid = torch.tensor([item['wid'] for item in batch])
        lexicon, lexicon_length = self.encode(transcr)

        return {'img':imgs, 'wid':wid, 'transcr': transcr, 'image_name':image_name, 'glyph': glyphs, 
                'lexicon': lexicon, 'lexicon_length': lexicon_length}
    

    def encode(self, text):
        """ convert a batch of text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) for s in text]
        batch_max_length  = max(length)+2
        # additional +2 for [GO] at the first step and [END] at the last step. batch_text is padded with [PAD] token after [END] token.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(self.dict['[PAD]'])
        batch_text[:, 0] = torch.LongTensor(len(text)).fill_(self.dict['[GO]'])
        for i, t in enumerate(text):
            text_new = list(t)
            text_new.append('[END]')
            text_new = [self.dict[char] if char in self.dict else len(self.dict) for char in text_new]
            batch_text[i][1:1 + len(text_new)] = torch.LongTensor(text_new)
        
        return (batch_text, torch.IntTensor(length))
    
    def decode(self, text_index):
        """ convert text-index into text-label. """
        text_index = text_index[:,1:]
        texts = []
        for index, t in enumerate(text_index):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            end_pos = text.find('[END]')
            text = text[:end_pos] if end_pos != -1 else text
            text = text.replace('[PAD]', 'P')
            texts.append(text)

        return texts
    