import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import Adam
import editdistance
import numpy as np

from hw_line_modules import HWLine_Module, init_net
from hw_line_dataset import BaseDataset, letters
from discriminator import GANLoss

CHANNEL_SIZE = 4
HIDDEN_SIZE = 256
OUTPUT_SIZE = len(letters) + 3  # +3 for the special tokens
MAX_LENGTH = 912

BATCH_SIZE = 4
EPOCH = 240
LR = 0.0001
BETA1 = 0.5
CKPT = ''

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
    if CKPT != '':
        model.load_state_dict(torch.load(CKPT))
    else:
        init_net(model, 'normal', 0.02, ['cuda:0'])
        # encoder_dict = {k.replace('features.features', 'features'):v for k,v in torch.load(pretrain_encoder).items() if 'features' in k}
        # miss, unexp = model.encoder.load_state_dict(encoder_dict)

    optimizer = Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))

    criterion = GANLoss().to("cuda")

    dataset = BaseDataset(train_image_path, train_label_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn_, num_workers=8, pin_memory=True)
    val_dataset = BaseDataset(test_image_path, test_label_path)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn_, num_workers=8, pin_memory=True)

    for epoch in range(EPOCH):
        # train
        model.train()
        loss_forradical = torch.tensor(0.0)
        losses = []
        for i, data in enumerate(loader):
            images = data["img"].to("cuda")
            lexicon, lexicon_length = data["lexicon"].to("cuda"), data["lexicon_length"].to("cuda")
            glyphs = data["glyphs"].to("cuda")
            word_idx = data["word_idx"]

            images = vae.encode(images).latent_dist.sample()
            images = images * 0.18215

            glyphs = vae.encode(glyphs).latent_dist.sample()
            glyphs = glyphs * 0.18215

            loss_forradical, pred = model(images, lexicon, lexicon_length, word_idx, glyphs)
            d_real_loss = criterion(pred, True)
            optimizer.zero_grad()
            loss = loss_forradical + d_real_loss
            loss.backward()
            optimizer.step()

            losses.append(loss_forradical.item())

            if i % 100 == 0:
                print(f"Epoch {epoch} step {i} loss: {np.mean(losses)}", flush=True)
        print(f"Epoch {epoch} loss: {np.mean(losses)}", flush=True)

        # save model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"ckpt/hw_line_ep_{epoch}.pt")
            print(f"Model saved as ckpt/hw_line_ep_{epoch}.pt")
        
        # validate
        if epoch % 5 == 0:
            model.eval()
            tdecs = []
            transcrs = []
            for i, data in enumerate(val_loader):
                images = data["img"].to("cuda")
                lexicon, lexicon_length = data["lexicon"].to("cuda"), data["lexicon_length"].to("cuda")
                transcr = data["transcr"]

                images = vae.encode(images).latent_dist.sample()
                images = images * 0.18215

                texts = torch.full(lexicon.shape, 2).to("cuda")
                texts[:, 0] = torch.LongTensor(lexicon.shape[0]).fill_(dataset.dict['[GO]'])

                pred_forradical, attention_maps = model.predict(images, texts)
                texts = dataset.decode(pred_forradical)
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



if __name__ == '__main__':
    main()