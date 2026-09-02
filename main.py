#!/usr/bin/env python3
import os
import torch
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed
from models import get_model_class

from argument_parser import (
    build_config_dict,
    parse_args,
)


_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    model_name = args.model
    model_class = get_model_class(model_name)
    config_dict = build_config_dict(args)

    config = Config(model=model_class, dataset=args.dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], model_name)(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    test_result = trainer.evaluate(test_data)
    logger.info('MODEL TEST')
    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))


if __name__ == '__main__':
    main()
