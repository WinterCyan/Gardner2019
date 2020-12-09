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
    print("creating model...")

    pretrained_densenet121 = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    pretrained_densenet121_dict = pretrained_densenet121.state_dict()
    model = ParamLENet().to(device)
    # init_paramlenet = ParamLENet()
    paramlenet_dict = model.state_dict()
    shared_weights = {k:v for k, v in pretrained_densenet121_dict.items() if k in paramlenet_dict}
    paramlenet_dict.update(shared_weights)
    model.load_state_dict(paramlenet_dict)
    model.to(device)

    lsc_loss_func = LSCLoss()
    refine_loss_func = RefineLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=default_lr, )
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)

    # optimizer = optim.Adam(model.parameters(), lr=default_lr)
    # scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    # ckp_path = 'modelsavings/checkpoint.pt'
    # print('loading model...')
    # model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)  # load_epoch: [1,150]
    # print("loaded model, epoch: ", load_epoch, ", training started...")

    print("loading data...")
    data = LEDataset(data_dir=cropped_imgs_dir)
    split_ratio = 0.2
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    training_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(training_data)/training_data_loader.batch_size))

    load_epoch = 0

    model.train()
    epochs = 150
    for epoch in range(load_epoch, epochs):
        print('----------------------training, EPOCH {}---------------------'.format(epoch+1))
        epoch_loss = 0
        epoch_start_time = time.time()
        for batch_idx, sample in enumerate(training_data_loader):
            # load data
            img_batch = sample["img"].to(device)
            print("loaded img_batch")
            gt_ambient_batch = sample["gt_ambient"].to(device)
            print("loaded gt_ambient_batch")
            gt_light_env_name_batch = sample["gt_light_env_name"]
            print("loaded gt_light_env_name_batch")
            # estimate
            torch.cuda.empty_cache()
            estimated_param_batch = model(img_batch)  # net_output: list of length 5: [d,l,s,c,a]; [N,3],[N,9],[N,3],[N,9],[N,3]
            print("network estimated param")
            estimated_d_batch = estimated_param_batch[0].to(device)  # shape: [N,9]
            estimated_l_batch = estimated_param_batch[1].to(device)  # shape: [N,9]
            estimated_s_batch = estimated_param_batch[2].to(device)  # shape: [N,3]
            estimated_c_batch = estimated_param_batch[3].to(device)  # shape: [N,9]
            estimated_a_batch = estimated_param_batch[4].to(device)  # shape: [N,3]
            # calculate loss
            lsc_loss = lsc_loss_func(gt_light_env_name_batch,
                                     estimated_l_batch,
                                     estimated_s_batch,
                                     estimated_c_batch,
                                     gt_ambient_batch,
                                     estimated_a_batch)
            epoch_loss += lsc_loss.item()
            optimizer.zero_grad()
            lsc_loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx+1) % 10 == 0:
                print("Epoch {:d}, Batch {:d}...".format(epoch+1, batch_idx+1))
            if batch_idx == batch_num-1:
                print("Epoch {:d}, Batch {:d}...".format(epoch+1, batch_idx+1))
        epoch_end_time = time.time()
        epoch_time = epoch_end_time-epoch_start_time
        print('----------------------EPOCH{}---------------------'.format(epoch+1))
        print("epoch mean loss: "+str(epoch_loss/batch_num))
        print("epoch time: "+str(epoch_time))
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        is_best = True if epoch == epochs-1 else False
        save_ckp(checkpoint, is_best, 'modelsavings', 'modelsavings')
        print('model saved, epoch: ', epoch+1)
        print('--------------------------------------------------')
