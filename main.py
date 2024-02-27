import hydra
from omegaconf import DictConfig, OmegaConf
from src import utils


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    pass

if __name__ == "__main__":
    main()
