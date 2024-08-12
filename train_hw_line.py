import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import Adam
import editdistance
import numpy as np

from hw_line_modules import HWLine_Module, init_net
from hw_line_dataset import BaseDataset, letters

CHANNEL_SIZE = 4
HIDDEN_SIZE = 256
OUTPUT_SIZE = len(letters) + 3  # +3 for the special tokens
MAX_LENGTH = 228

BATCH_SIZE = 64
EPOCH = 240
LR = 0.001
BETA1 = 0.5

CKPT = ''
train_image_path = '/home/SSD/yutao/IAM-lines/crop/images/train'
train_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-train.txt'
test_image_path = '/home/SSD/yutao/IAM-lines/crop/images/test'
test_label_path = '/home/SSD/yutao/IAM-lines/crop/IAM-test.txt'

def main():
    vae = AutoencoderKL.from_pretrained("./diffusion", subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to("cuda")
    model = HWLine_Module(CHANNEL_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, max_length=MAX_LENGTH).to("cuda")
    if CKPT != '':
        model.load_state_dict(torch.load(CKPT))
    else:
        init_net(model, 'normal', 0.02, ['cuda:0'])
    optimizer = Adam(model.parameters(), lr=LR, betas=(BETA1, 0.999))

    dataset = BaseDataset(train_image_path, train_label_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn_, num_workers=8, pin_memory=True)
    val_dataset = BaseDataset(test_image_path, test_label_path)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn_, num_workers=8, pin_memory=True)

    for epoch in range(EPOCH):
        # train
        model.train()
        loss_forradical = torch.tensor(0.0)
        losses = []
        for data in loader:
            images = data["img"].to("cuda")
            lexicon, lexicon_length = data["lexicon"].to("cuda"), data["lexicon_length"].to("cuda")

            images = vae.encode(images).latent_dist.sample()
            images = images * 0.18215

            loss_forradical = model(images, lexicon, lexicon_length)
            optimizer.zero_grad()
            loss_forradical.backward()
            optimizer.step()

            losses.append(loss_forradical.item())
        print(f"Epoch {epoch} loss: {np.mean(losses)}")

        # save model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"ckpt/hw_line_ep_{epoch}.pth")
            print(f"Model saved as ckpt/hw_line_ep_{epoch}.pth")
        
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