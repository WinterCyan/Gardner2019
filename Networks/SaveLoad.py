import torch
import shutil


def save_ckp(state, is_best, ckp_dir, best_model_dir):
    fpath = ckp_dir+'checkpoint.pt'
    # torch.save(state, fpath)
    torch.save(state, fpath)
    if is_best:
        best_fpath = best_model_dir+'best_model.pth'
        shutil.copyfile(fpath, best_fpath)


def load_ckp(fpath, model, optimizer):
    ckp = torch.load(fpath)
    model.load_state_dict(ckp['state_dict'])
    optimizer.load_state_dict(ckp['optimizer'])
    return model, optimizer, ckp['epoch']


# loading:
# model = MyModel(*args, **kwargs)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# ckp_path = "path/to/checkpoint/checkpoint.pt"
# model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
