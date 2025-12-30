import logging

import hydra
from omegaconf import OmegaConf

from ragfit.processing.pipeline import DataPipeline

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs/my_configs", config_name="processing-retrieval")
def main(args):
    logger.info(OmegaConf.to_yaml(args))

    pipeline = DataPipeline(**args)
    pipeline.process()


if __name__ == "__main__":
    main()
