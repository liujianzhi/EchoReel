import os
import argparse
from omegaconf import OmegaConf

from lvdm.data.ucf import ucf
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from lvdm.data.ucf import ucf
from lvdm.models.ddpm3d import LatentDiffusion
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from lvdm.utils.common_utils import instantiate_from_config
from torch.utils.data import DataLoader, Dataset
import setproctitle
from data import create_dataset
def get_parser():
    parser = argparse.ArgumentParser()
    """ Base args """
    parser.add_argument('--name', type=str, default='main', help='experiment identifier')
    parser.add_argument('--savedir', type=str, default='logs', help='path to save checkpoints and logs')
    parser.add_argument('--savevideo', type=str, default='videos', help='path to save videos')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='experiment mode to run')

    """ Date args """
    parser.add_argument('--train_dataset', type=str, default='') # webvid
    parser.add_argument('--test_dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)

    """ Model args """
    parser.add_argument("--config", type=str, help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument('--resume', type=str, default='')

    """ Args about Training """
    parser.add_argument('--nodes', type=int, default=1, help='nodes')
    parser.add_argument('--devices', type=int, default=8, help='e.g., gpu number')
    return parser.parse_args()

def get_nested_attr(obj, attr_string):
    attrs = attr_string.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def main():
    args = get_parser()
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    setproctitle.setproctitle(args.name)
    config.name = args.name
    config.savedir = os.path.join(args.savedir, args.name)
    config.batch_size = args.batch_size
    config.model.params.name = config.name
    config.model.params.world_size = args.devices
    
    checkpoint_callback = ModelCheckpoint(
        dirpath                   =     os.path.join(config.savedir, 'checkpoints'),
        filename                  =     '{step}', # -{epoch:02d}
        monitor                   =     'step',
        save_last                 =     False,
        save_top_k                =     -1,
        verbose                   =     True,
        every_n_train_steps       =     1500,
        save_on_train_epoch_end   =     True,
    )

    strategy = DeepSpeedStrategy(
        stage                     =     2, 
        offload_optimizer         =     True, 
        load_full_weights         =     True,
    )

    trainer = pl.Trainer(
        default_root_dir          =     config.savedir,
        callbacks                 =     [checkpoint_callback, ], # ModelSummary(2)
        accelerator               =     'cuda',
        benchmark                 =     True,
        num_nodes                 =     args.nodes,
        devices                   =     args.devices,
        log_every_n_steps         =     1,
        precision                 =     16,
        max_epochs                =     config.num_train_epochs,
        strategy                  =     strategy,
        sync_batchnorm            =     True,
        val_check_interval        =     200,
    )
    config.model.params.global_rank = trainer.global_rank

    if args.mode == "train":
        train_dataloader = DataLoader(create_dataset(args.train_dataset, False), shuffle=True, batch_size=config.batch_size, num_workers=6)
        test_dataloader = DataLoader(create_dataset(args.test_dataset, True), shuffle=False, batch_size=config.batch_size, num_workers=6)
    else:
        test_dataloader = DataLoader(create_dataset(args.test_dataset, False), shuffle=False, batch_size=config.batch_size, num_workers=6)

    trainer_model = instantiate_from_config(config.model)
    
    for name, parameter in trainer_model.named_parameters():
        if 'EchoReel' in name:
            parameter.requires_grad = True
            obj = get_nested_attr(trainer_model, name)
        else:
            parameter.requires_grad = False
    

    if args.mode =='train':
        trainer_model.load_state_dict(torch.load('models/t2v/model.ckpt', map_location='cpu'), strict=False)
    else:
        d = torch.load(args.resume, map_location='cpu')
        d_con = {}
        for t in d['module']:
            d_con[t[t.find('.') + 1:]] = d['module'][t]
        trainer_model.load_state_dict(d_con, strict=True)

    if args.mode == 'train':
    ### training
        trainer.fit(
            model                     =     trainer_model,
            train_dataloaders         =     train_dataloader,
            val_dataloaders           =     test_dataloader,
            ckpt_path                 =     None if not os.path.exists(args.resume) else args.resume,
        )
    elif args.mode == 'test':
        assert os.path.exists(args.resume), "resume path does not exist"
        trainer.test(
            model                     =     trainer_model,
            dataloaders               =     test_dataloader,
        )





if __name__ == "__main__":
    main()