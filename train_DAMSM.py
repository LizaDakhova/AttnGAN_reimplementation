from __future__ import print_function

from utils import mkdir_p
from utils import build_super_images
from losses import sent_loss, words_loss
from config import cfg

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import time
import datetime
import dateutil.tz
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


UPDATE_INTERVAL = 10


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()

    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        words_features, sent_code = cnn_model(imgs[-1])
        nef, att_sze = words_features.size(1), words_features.size(2)
        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data

        loss.backward()

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0 / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1 / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()

    # Attention Maps
    img_set, _ = build_super_images(
        imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = f"{image_dir}/attention_maps{epoch:03d}.png"
        im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, labels, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step

    return s_cur_loss, w_cur_loss


def build_models(n_words, batch_size):
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = torch.arange(batch_size)
    history = {"s_loss": [], "w_loss": []}
    lr = cfg.TRAIN.ENCODER_LR
    start_epoch = 0

    if cfg.STATE_DICT is not None:
        text_encoder.load_state_dict(cfg.STATE_DICT["DAMSM"]["text_encoder"])
        image_encoder.load_state_dict(cfg.STATE_DICT["DAMSM"]["image_encoder"])
        history = cfg.STATE_DICT["DAMSM"]["history"]
        lr = cfg.STATE_DICT["DAMSM"]["lr"]
        start_epoch = cfg.STATE_DICT["DAMSM"]["epoch"]

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, history, lr, start_epoch


def pipeline():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    state_dir = os.path.join(cfg.OUTPUT_DIR, f"State_DAMSM_{timestamp}")
    mkdir_p(state_dir)
    state_path = os.path.join(state_dir, f"state_{timestamp}.pth")
    image_dir = os.path.join(cfg.OUTPUT_DIR, f"Image_DAMSM_{timestamp}")
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    batch_size = cfg.TRAIN.BATCH_SIZE

    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    train_dataset = TextDataset(
        cfg.DATA_DIR, 'train',
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    val_dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    text_encoder, image_encoder, labels, history, lr, start_epoch = \
        build_models(train_dataset.n_words, batch_size)

    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            train(train_dataloader, image_encoder, text_encoder,
                  batch_size, labels, optimizer, epoch,
                  train_dataset.ixtoword, image_dir)

            print('-' * 89)

            if len(val_dataloader):
                s_loss, w_loss = evaluate(val_dataloader, image_encoder, text_encoder, labels, batch_size)
                history["s_loss"].append(float(s_loss.data))
                history["w_loss"].append(float(w_loss))
                print('| end epoch {:3d} | valid loss ' '{:5.2f} {:5.2f} | lr {:.5f}|'.format(epoch, s_loss, w_loss, lr))

            print('-' * 89)

            if lr > cfg.TRAIN.ENCODER_LR / 10.:
                lr *= 0.98

            if cfg.CUDA:
                print("Peak memory usage by Pytorch tensors:",
                      round(torch.cuda.max_memory_allocated(f"cuda:{cfg.GPU_ID}") / 1024 ** 2, 3), "Mb")

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch == cfg.TRAIN.MAX_EPOCH:
                state_dict = {
                    "DAMSM": {
                        "image_encoder": image_encoder.state_dict(),
                        "text_encoder": text_encoder.state_dict(),
                        "history": history,
                        "lr": lr,
                        "epoch": epoch,
                    }
                }
                torch.save(state_dict, state_path)
                print('Save image_encoder and text_encoder models.')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
