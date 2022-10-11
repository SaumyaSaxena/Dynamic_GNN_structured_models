import os
from pathlib import Path
import logging
import numpy as np
import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from Dynamic_GNN_structured_models.datasets import *
from Dynamic_GNN_structured_models.learning import *

logger = logging.getLogger(__name__)

@hydra.main(config_path='../cfg/', config_name='train/train_forward_model.yaml')
def main(cfg):
    if 'save_path_prefix' in cfg and len(cfg['save_path_prefix']) > 0:
        hydra_dir = Path(f"{os.getcwd()}/{cfg['save_path_prefix']}")
    else:
        hydra_dir = Path(os.getcwd())
    
    checkpoint_path = hydra_dir / 'checkpoints'
    logger.info(f"====> Using checkpoint path: {checkpoint_path}")

    callbacks = []
    logger_type = cfg['train']['logger_type']
    if logger_type == 'wandb':
        wandb_logger = WandbLogger(name=cfg['tag'], config=cfg['train'], **cfg['wandb']['logger'])
        wandb_logger.experiment # hack to explicitly trigger wandb init
        callbacks.append(WandbUploadCallback(
            checkpoint_path, hydra_dir, cfg['wandb']['saver']['upload']
        ))
    elif logger_type == 'tb':
        wandb_logger = TensorBoardLogger(hydra_dir / 'logs')
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")
    
    if cfg['train'].get('early_stopping', None) is not None:
        if cfg['train']['early_stopping']['use']:
            callbacks.append(EarlyStopping(
                monitor=cfg['wandb']['saver']['monitor'],
                **cfg['train']['early_stopping']['params']
            ))

    callbacks.append(ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=str(checkpoint_path),
        filename='{epoch:06d}-{val_loss:.4f}',
        save_top_k=cfg['wandb']['saver'].get('save_top_k', 20),
        save_last=True
    ))
    
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    logger.info(f"Using GPU: {cfg.get('gpu', 0)}")
    # Make trainer
    trainer = Trainer(
        gpus=[cfg.get('gpu', 0)],
        fast_dev_run=cfg['debug'],
        track_grad_norm=2,
        logger=wandb_logger,
        max_epochs=cfg['train']['max_epochs'],
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=cfg['train']['log_every_n_steps']
    )

    # Loading the datasets
    dataset = eval(cfg['data']['dataset_class'])()

    train_size = int(len(dataset) * cfg['train']['train_val_split'])
    val_size = len(dataset) - train_size
    
    train_data, val_data = torch.utils.data.random_split(dataset, (train_size, val_size))

    cfg_data_dict = dataset.get_config()
    cfg_train_dict = {k: v for k, v in cfg['train'].items()}
    cfg_model_dict = {k: v for k, v in cfg['models'][cfg['train']['model_name']].items()}
    cfg_dict = {**cfg_data_dict, **cfg_train_dict, **cfg_model_dict}

    model_cls = eval(cfg['train']['model_name'])
    model = model_cls(
        train_ds_gen=lambda : train_data, 
        val_ds_gen=lambda: val_data, 
        cfg_dict=cfg_dict
        )
    
    if cfg.get('use_checkpoint', False):
        ckpt_path = cfg['checkpoint_props']['ckpt_path']
        ckpt_name = cfg['checkpoint_props']['checkpoint']
        logger.info(f"Will load ckpt_file: {ckpt_path}")
        checkpoint = torch.load(ckpt_path + ckpt_name)
        model.load_state_dict(checkpoint['state_dict'])

    trainer.fit(model)

if __name__ == "__main__":
    main()