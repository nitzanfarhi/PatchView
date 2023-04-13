import logging

import torch
from code_model import Model
from code_training import MODEL_CLASSES, evaluate, load_model, parse_args, test, train
from message_dataset import MessageDataset
from transformers import RobertaTokenizer, AdamW
import argparse

from misc import set_seed
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    args = parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARN)
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    # do_checkpoint(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name or args.model_name_or_path,
        cache_dir=args.cache_dir or None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir or None,
    )
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = model_class(config)

    model = Model(model, config, tokenizer, args)

    logger.info("Training/evaluation parameters %s", args)

    eval_dataset = MessageDataset(tokenizer, args, 'val') 

    if args.do_train:
        from events_training import train as mytrain
        mconfig = { 'batch_size': 8, "optimizer":AdamW, "lr":0.1}
        model.to(args.device)
        mytrain(model,mconfig, args, MessageDataset(tokenizer,args, 'train'), MessageDataset(tokenizer,args, 'val'), name ='message')

    if args.do_eval:
        load_model(args, model, 'message')
        result = evaluate(args, model, tokenizer, eval_dataset)
        logger.warning("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.warning("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        load_model(args, model, 'message')
        test_dataset = MessageDataset(tokenizer, args, 'test')
        test(args, model, tokenizer, test_dataset, 'message')


if __name__ == "__main__":
    main()
