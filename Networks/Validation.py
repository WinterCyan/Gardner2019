import time
from torch.utils.data import DataLoader, random_split
from Dataset.Dataset import *
from Networks.Network import *
from Networks.Losses import *
from torch.optim.lr_scheduler import StepLR
from Networks.SaveLoad import *


if __name__ == '__main__':
    device = 'cuda:0'
    default_lr = 0.001
    print("loading neural network...")
    # pretrained_densenet121 = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    # pretrained_densenet121_dict = pretrained_densenet121.state_dict()
    model = ParamLENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=default_lr)
    ckp_path = '../modelsavings/checkpoint.pt'
    model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)
    # init_paramlenet = ParamLENet()

    # paramlenet_dict = model.state_dict()
    # shared_weights = {k:v for k, v in pretrained_densenet121_dict.items() if k in paramlenet_dict}
    # paramlenet_dict.update(shared_weights)
    # model.load_state_dict(paramlenet_dict)
    # model.to(device)
    lsc_loss_func = LSCLoss()

    print("loading data...")
    data = LEDataset(data_dir=cropped_imgs_dir)
    split_ratio = 0.1
    _, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    validation_data_loader = DataLoader(validation_data, batch_size=16, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(validation_data)/validation_data_loader.batch_size))
    print("loaded {:d} images, validating neural network...".format(len(validation_data)))

    model.eval()
    val_loss = 0
    val_batch_num = 0
    for batch_idx, sample in enumerate(validation_data_loader):
        # load data
        img_batch = sample["img"].to(device)
        gt_ambient_batch = sample["gt_ambient"].to(device)
        gt_light_env_batch = sample["gt_light_env"].to(device)
        # estimate
        estimated_param_batch = model(img_batch)  # net_output: list of length 5: [d,l,s,c,a]; [N,3],[N,9],[N,3],[N,9],[N,3]
        estimated_d_batch = estimated_param_batch[0].to(device)  # shape: [N,9]
        estimated_l_batch = estimated_param_batch[1].to(device)  # shape: [N,9]
        estimated_s_batch = estimated_param_batch[2].to(device)  # shape: [N,3]
        estimated_c_batch = estimated_param_batch[3].to(device)  # shape: [N,9]
        estimated_a_batch = estimated_param_batch[4].to(device)  # shape: [N,3]
        # calculate loss
        lsc_loss = lsc_loss_func(gt_light_env_batch,
                                 estimated_l_batch,
                                 estimated_s_batch,
                                 estimated_c_batch,
                                 gt_ambient_batch,
                                 estimated_a_batch)
        val_loss += lsc_loss.item()
        val_batch_num += 1
    val_loss /= val_batch_num
    print()
    print('----------------------------------------------')
    print("DONE.")
    print("Validation batch: {:d}, validation loss: {:.4f}".format(val_batch_num, val_loss))
    print("Similarity of lighting: {:.2f}%.".format((1.0-val_loss)*100))