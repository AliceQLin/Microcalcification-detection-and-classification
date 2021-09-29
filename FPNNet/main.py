from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import NBIDataset
from torchvision import transforms
from model import rcnn_model
from torch.utils.data import DataLoader, Subset
from demo import visualize
import os


def get_transform(train):
    all_transforms = []
    if train:
        all_transforms.append(transforms.RandomHorizontalFlip(0.5))
    all_transforms.append(transforms.ToTensor())
    return transforms.Compose(all_transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = rcnn_model
    model.to(device)
    
#    pretrained_model_path = os.path.join("./models", 'model_epoch-50.pt')
#    net_state_dict = torch.load(pretrained_model_path)
#    model.load_state_dict(net_state_dict)
    
    start_epoch = 0

    # use our dataset and defined transformations
    root = "./MC_dataset"
    dataset_train = NBIDataset(os.path.join(root, "train"), get_transform(train=True))

    # define training and validation data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)

    # construct an optimizer and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    num_epochs = 2000

    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join("./models", 'model_epoch-%d.pt' % epoch))

if __name__ == '__main__':
    
    main()