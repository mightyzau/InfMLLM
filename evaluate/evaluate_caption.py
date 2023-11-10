import os, sys 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import argparse
import itertools
import json
import re
from functools import partial
from PIL import Image
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import torch

from infmllm.processors.processors import Blip2ImageEvalProcessor
from infmllm.models import build_model


ds_collections = {
    "coco_test": {
        "img_dir": "datasets/COCO/",
        "test": "datasets/COCO/annotations/coco_karpathy_test_gt_converted.json",
    },
    "coco_val": {
        "img_dir": "datasets/COCO/",
        "test": "datasets/COCO/annotations/coco_karpathy_val_gt_converted.json",
    },
    "flickr_test": {
        "img_dir": "datasets/flickr30k/flickr30k_images/",
        "test": "datasets/flickr30k/flickr30k_karpathy_test.json",
    },
    "nocaps_val": {
        "img_dir": "datasets/nocaps/images/",
        "test": "datasets/nocaps/annotations/nocaps_val_converted.json",
    }
}

def collate_fn(batches):
    images = torch.stack([_['image'] for _ in batches], dim=0)
    prompts = [_['prompt'] for _ in batches]
    image_ids = [_['image_id'] for _ in batches]

    return {
        'image': images,
        'prompts': prompts,
        'image_ids': image_ids,
    }

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, test, prompt, vis_processor):
        self.img_dir = img_dir
        self.prompt = prompt
        self.images = json.load(open(test))['images']
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info['id']
        image_path = os.path.join(self.img_dir, info['image'])
        assert os.path.isfile(image_path), "{} not exist.".format(image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        return {
            'image': image,
            'image_id': image_id,
            'prompt': self.prompt,
        }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    
def load_checkpoint(model, ckpt_file):
    if os.path.isfile(ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid: {}".format(ckpt_file))
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)

    print("Missing keys {}".format(msg.missing_keys))
    print("load checkpoint from %s" % ckpt_file)
    return msg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument('--model_type', type=str, default="infmllm_inference_llama")
    # for vit model
    parser.add_argument('--vit_model', type=str, default="eva_clip_g")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--vit_lora_config', type=str, default="")
    parser.add_argument('--vit_adapter_convpass', action='store_true')
    # for vision adapter
    parser.add_argument('--vision_adapter', type=str, default="pooler")
    parser.add_argument('--pool_out_size', type=int, default=16)
    # for lm model
    parser.add_argument('--lm_model', type=str, default="pretrain_models/llama-2-7b-chat-hf/")
    parser.add_argument('--lm_tokenizer', type=str, default="pretrain_models/llama-2-7b-chat-hf/")
    parser.add_argument('--lm_lora_config', type=str, default="")
    parser.add_argument('--apply_lemmatizer', type=str, default="False")
    parser.add_argument('--precision', type=str, default="amp_bf16")

    parser.add_argument('--prompt', type=str)

    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model = build_model(
        model_type=args.model_type,
        vit_model=args.vit_model,
        img_size=args.image_size,
        vision_adapter=args.vision_adapter,
        lm_model=args.lm_model,
        lm_tokenizer=args.lm_tokenizer,
        precision=args.precision,
        args=args
    )
    model = model.cuda().eval()
    load_checkpoint(model, args.checkpoint)

    ciders_list = []
    for dataset_name in args.dataset.split(','):
        dataset_name = dataset_name.strip()

        image_processor = Blip2ImageEvalProcessor(image_size=args.image_size)
        dataset = CaptionDataset(
                img_dir=ds_collections[dataset_name]['img_dir'],
                test=ds_collections[dataset_name]['test'],
                prompt=args.prompt,
                vis_processor=image_processor)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        image_ids = []
        captions = []
        for _, samples in tqdm(enumerate(dataloader)):
            samples['image'] = samples['image'].cuda()
            answers = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=30,
                min_length=8
            )
            image_ids.extend(samples['image_ids'])
            captions.extend(answers)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [
            _ for _ in itertools.chain.from_iterable(merged_captions)
        ]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")

            results = []
            for image_id, caption in zip(merged_ids, merged_captions):
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
            #time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            #results_file = f'{dataset_name}_{time_prefix}.json'

            results_file = os.path.splitext(args.checkpoint)[0] + f'_{dataset_name}_{args.image_size}.json'
            json.dump(results, open(results_file, 'w'))

            coco = COCO(ds_collections[dataset_name]['test'])
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            print(coco_eval.eval.items())

            ciders_list.append(coco_eval.eval['CIDEr'])
        torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        print()
        print('*' * 50)
        file = os.path.splitext(args.checkpoint)[0] + '_eval.txt'
        with open(file, 'a') as f:
            for dataset_name, cider in zip(args.dataset.split(','), ciders_list):
                print('  {}: {}'.format(dataset_name, cider))
                f.write('{}: {}\n'.format(dataset_name, cider))
