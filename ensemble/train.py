import torch
from torch.optim import AdamW
from torch import nn
from tqdm.auto import tqdm
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from datasets import load_metric
from torch import optim

metric = load_metric("mean_iou")

from model import get_model, get_palette
from dataset import get_dataloader
from metrics import iou_pytorch

root_dir = '/home/winky/PycharmProjects/Trian/data'
batch_size = 3

torch.cuda.empty_cache()
device = torch.device("cuda")  # if torch.cuda.is_available() else torch.device("cpu")
model, feature_extractor = get_model()
# model = torch.load('checkpoints/30_7_0.8171499967575073_model.pth')
model.to(device)
train_dataloader, val_dataloader = get_dataloader(root_dir, feature_extractor, train_batch_size=batch_size,
                                                  val_batch_size=batch_size)
id2label, label2id, id2color = get_palette()

optimizer = AdamW(model.parameters(), lr=0.00006)
num_epochs = 100
num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer, num_warmup_steps=int(num_training_steps * 0.03), num_training_steps=num_training_steps,
# )
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8 * len(train_dataloader), eta_min=0.00000006)
# progress_bar = tqdm(range(num_training_steps))


best_val_loss = 1000
last_val_lost = 0

for epoch in range(num_epochs):
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # # evaluate
        # with torch.no_grad():
        #     upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
        #                                                  align_corners=False)
        #     predicted = upsampled_logits.argmax(dim=1)
        #
        #     # note that the metric expects predictions + labels as numpy arrays
        #     metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        #
        # # let's print loss and metrics every 100 batches
        # if idx % 100 == 0:
        #     metrics = metric.compute(num_labels=len(id2label),
        #                              ignore_index=255,
        #                              reduce_labels=False,  # we've already reduced the labels before)
        #                              )
        #
        #     print("Loss:", loss.item())
        #     print("Mean_iou:", metrics["mean_iou"])
        #     print("Mean accuracy:", metrics["mean_accuracy"])
    lr = lr_scheduler.get_last_lr()

    m_ioyu = []
    model.eval()
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        # evaluate
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            # metric.add_batch(predictions=predicted, references=labels)
            iou = iou_pytorch(outputs=predicted, labels=labels)
            m_ioyu.append(iou)
    miou = sum(m_ioyu) / len(m_ioyu)
    torch.save(model, f'checkpoints/mit_{epoch}_{miou.item()}_model.pth')
    print(f'Epoch: {epoch}  LR: {lr}')
    print('miou:', miou.item())
    # metrics = metric.compute(num_labels=len(id2label),
    #                          ignore_index=255,
    #                          reduce_labels=False,  # we've already reduced the labels before)
    #                          )
# 2803 - 7478
# 3033 - 7520
#
# print("Mean_iou:", metrics["mean_iou"])
# print("Mean accuracy:", metrics["mean_accuracy"])
