from omegaconf import DictConfig, OmegaConf
import hydra
from dataloader import BirdModule
from pytorch_lightning.trainer import Trainer
from birdmodels.BirdModel import BirdModel
from lightning.pytorch.loggers import WandbLogger

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #*******logging*********
    wandb_logger = WandbLogger(entity=cfg.wandb_entity,
                               project="Birdcleff")

    datam = BirdModule(cfg, n_split=1)
    datam.setup()
    train_loader = datam.train_dataloader()
    val_loader = datam.val_dataloader()

    model=BirdModel(cfg,fold=1,num_classes=datam.get_num_classes())

    trainer=Trainer(logger=wandb_logger,
                accelerator='gpu',
                devices=1,
                strategy='auto',
                max_epochs=cfg.epochs,
                check_val_every_n_epoch=1,
                enable_progress_bar=True)
    trainer.fit(model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)

if __name__ == "__main__":
    train()