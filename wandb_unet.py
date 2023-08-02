import cv2
import torch
import torch.nn as nn
import unet as Model
from torch.utils.data import Dataset, DataLoader
from torch import optim
from glob import glob
import wandb
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['WANDB_NOTEBOOK_NAME'] = 'wandb_unet.py'

wandb.login()

num_trails = 1
sweep_config = {'method': 'bayes'} # Try Bayesian Optimization
metric = {'name': 'loss','goal': 'minimize'}
sweep_config['metric'] = metric

parameters_dict = {
    'epochs': {'value': 200}
  }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
  # 'learning_rate': {'distribution': 'uniform','min': 0.0001,'max': 0.001},
  # 'beta1': {'distribution': 'uniform','min': 0.6,'max': 0.99},
  # 'beta2': {'distribution': 'uniform','min': 0.7,'max': 1},
  'batch_size': {'value': 4},
  'learning_rate': {'value': 0.00018522214761659513},
  'beta1': {'value': 0.7408185044206802},
  'beta2': {'value': 0.7117526519722864},
  })


class SegData(Dataset):
  def __init__(self, split):
    self.img = glob("dataset1/images_prepped_{}/*.png".format(split), recursive=True)
    self.mask = glob("dataset1/annotations_prepped_{}/*.png".format(split), recursive=True)
  def __len__(self):
    return len(self.img)
  def __getitem__(self, ix):
    img = cv2.imread(self.img[ix], cv2.COLOR_BGR2RGB)
    mask = cv2.imread(self.mask[ix], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (572, 572))
    mask = cv2.resize(mask, (388, 388))
    image = torch.tensor(img).permute(2,0,1).to(device)
    mask = torch.tensor(mask).reshape(388,388).to(device)
    image = image.float() / 255.
    mask = mask.long()
    return image, mask
  

def build_dataset(batch_size):
  trn_ds = SegData('train')
  val_ds = SegData('test')
  trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
  val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)
  return trn_dl, val_dl

def build_network():
  model = Model.UNet().to(device) 
  return model

def build_optimizer(network, learning_rate, beta1, beta2):
  optimizer = optim.AdamW(network.parameters(), lr=learning_rate, betas=(beta1, beta2))
  return optimizer

ce =  nn.CrossEntropyLoss()
def network_loss(pred, target):
  loss = ce(pred, target)
  acc = (torch.max(pred, 1)[1] == target).float().mean()
  return loss, acc

def train_epoch(network, loader, optimizer):
  cumu_loss = 0
  cumu_acc = 0
  for _, (image, mask) in enumerate(loader):
    optimizer.zero_grad()
    network.train()
    _mask = network(image)
    loss, acc = network_loss(_mask, mask)
    cumu_loss += loss.item()
    cumu_acc += acc
    loss.backward()
    optimizer.step() 
    wandb.log({"batch loss": loss.item(), "loss": loss.item(), "accuracy": acc})
  return cumu_loss / len(loader), cumu_acc / len(loader), _mask[0], mask[0], image[0]

def validate_epoch(network, loader):
  cumu_loss = 0
  cumu_acc = 0
  for _, (image, mask) in enumerate(loader):
    network.eval()
    _mask = network(image)
    loss, acc = network_loss(_mask, mask)
    cumu_loss += loss.item()
    cumu_acc += acc
    wandb.log({"val batch loss": loss.item(), "val loss": loss.item(), "val accuracy": acc})
  return cumu_loss / len(loader), cumu_acc / len(loader)

def train(config=None):
  with wandb.init(config=config):
    config = wandb.config
    table = wandb.Table(columns=["Epoch", "Loss", "Acc", "Prediction", "Truth", "Image"])
    trn_dl, _ = build_dataset(config.batch_size)
    network = build_network()
    optimizer = build_optimizer(network, config.learning_rate, config.beta1, config.beta2)
    for epoch in range(config.epochs):
      avg_loss, avg_acc, pred, mask, image = train_epoch(network, trn_dl, optimizer)
      wandb.log({"avg loss": avg_loss, "avg acc": avg_acc, "epoch": epoch})
      # avg_loss, avg_acc = validate_epoch(network, val_dl)
      # wandb.log({"avg val loss": avg_loss, "avg val acc": avg_acc, "epoch": epoch})
    table.add_data(epoch+1, avg_loss, avg_acc, wandb.Image(torch.argmax(pred, dim=0).squeeze().cpu().float()), wandb.Image(mask.float()), wandb.Image((image*255).float()))
    wandb.log({"result": table})
    torch.save(network.state_dict(), f'./weights/model_{avg_loss}.pth')
    del network

sweep_id = wandb.sweep(sweep_config, project="U-Net")
wandb.agent(sweep_id, train, count=num_trails)