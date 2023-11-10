import os, sys 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import itertools
import json
import re
from PIL import Image
import time

import torch
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from bigmodelvis import Visualization

from infmllm.processors.processors import Blip2ImageEvalProcessor, BlipQuestionProcessor
from evaluate.vqa_metric import VQA, VQAEval
from infmllm.models import build_model


ds_collections = {
    'refcoco_val': {
        'vis_root': 'datasets/refcoco/',
        'test': 'datasets/refcoco/val.json'},
    'refcoco_testA': {
        'vis_root': 'datasets/refcoco/',
        'test': 'datasets/refcoco/testA.json'},
    'refcoco_testB': {
        'vis_root': 'datasets/refcoco/',
        'test': 'datasets/refcoco/testB.json'},

    'refcoco+_val': {
        'vis_root': 'datasets/refcoco+/',
        'test': 'datasets/refcoco+/val.json'},
    'refcoco+_testA': {
        'vis_root': 'datasets/refcoco+/',
        'test': 'datasets/refcoco+/testA.json'},
    'refcoco+_testB': {
        'vis_root': 'datasets/refcoco+/',
        'test': 'datasets/refcoco+/testB.json'},

    'refcocog_val': {
        'vis_root': 'datasets/refcocog/',
        'test': 'datasets/refcocog/val.json'},
    'refcocog_test': {
        'vis_root': 'datasets/refcocog/',
        'test': 'datasets/refcocog/test.json'},
}


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def collate_fn(batches):
    images = torch.stack([_['image'] for _ in batches], dim=0)
    prompts = [_['prompt'] for _ in batches]
    bboxes = [_['bbox'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    box_ids = [_['box_id'] for _ in batches]

    return {
        'image': images,
        'prompts': prompts,
        'bboxes': bboxes,
        'hws': hws,
        'box_ids': box_ids,
    }

class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, vis_root, test, prompt, vis_processor):
        self.datas = []
        datas = json.load(open(test, 'r'))
        for d in datas:
            img_name = d['img_name']
            if 'train' in img_name:
                img_name = os.path.join(vis_root, 'train2014', img_name)
            else:
                img_name = os.path.join(vis_root, 'val2014', img_name)
            assert os.path.isfile(img_name)
            bbox = d['bbox']
            for sent in d['sentences']:
                self.datas.append({
                    'image': img_name,
                    'sent': sent['sent'],
                    'bbox': bbox,
                    "box_id": "{}_{}".format(d['segment_id'], sent['sent_id'])
                })

        print('Total {} samples.'.format(len(self.datas)))
        self.prompt = prompt
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        image = Image.open(data['image']).convert("RGB")
        w, h = image.size
        image = self.vis_processor(image)

        text = data['sent']
        bbox = data['bbox']
        if bbox[2] > w or bbox[3] > h:
            print('bbox {} outsize of w,h {}'.format(bbox, (w, h)))

        prompt = self.prompt.format(text)

        return {
            'image': image,
            'prompt': prompt,
            'bbox': bbox,
            'hw': (h, w),
            'box_id': data['box_id']
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
    parser.add_argument('--lm_custom', type=str, default="")
    parser.add_argument('--lm_lora_config', type=str, default="")
    parser.add_argument('--apply_lemmatizer', type=str, default="False")
    parser.add_argument('--precision', type=str, default="amp_bf16")

    parser.add_argument('--apply_grounding_refine', action='store_true')
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
    if torch.distributed.get_rank() == 0:
        Visualization(model).structure_graph()

    model = model.cuda().eval()

    load_checkpoint(model, args.checkpoint)

    precision_list = []
    for dataset_name in args.dataset.split(','):
        dataset_name = dataset_name.strip()

        image_processor = Blip2ImageEvalProcessor(image_size=args.image_size)
        dataset = RefCOCODataset(
                    vis_root=ds_collections[dataset_name]['vis_root'],
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

        outputs = []
        for _, samples in tqdm(enumerate(dataloader)):
            samples['image'] = samples['image'].cuda()
            answers = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=5,
                max_length=30,
                min_length=1
            )
            pred_bboxes = []
            for answer, (h, w) in zip(answers, samples['hws']):
                try:
                    x1, y1, x2, y2 = re.search(r'(0.\d+),(0.\d+).*(0.\d+),(0.\d+)', answer).groups()
                    predict_bbox = (float(x1) * w, float(y1) * h, float(x2) * w, float(y2) * h)
                except:
                    predict_bbox = (0., 0., 0., 0.)
                pred_bboxes.append(predict_bbox)

            bboxes = samples['bboxes']
            box_ids = samples['box_ids']

            for bbox, pred_bbox, box_id in zip(bboxes, pred_bboxes, box_ids):
                outputs.append({
                    'pred_bbox': pred_bbox,
                    'gt_bbox': bbox,
                    'box_id': box_id,
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")

            results_file = os.path.splitext(args.checkpoint)[0] + f'_{dataset_name}_{args.image_size}.json'
            json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

            correct = total_cnt = 0
            for i, output in enumerate(merged_outputs):
                target_bbox = torch.tensor(output['gt_bbox'], dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(output['pred_bbox'], dtype=torch.float32).view(-1, 4)
                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                total_cnt += 1
                if iou >= 0.5:
                    correct += 1

            print(f"Evaluating {dataset_name} ...")
            print(f'Precision @ 1: {correct / total_cnt} \n')
            precision_list.append(correct / total_cnt)
        torch.distributed.barrier()
    
    if torch.distributed.get_rank() == 0:
        print()
        print('*' * 50)
        
        file = os.path.splitext(args.checkpoint)[0] + '_eval.txt'
        with open(file, 'a') as f:
            for dataset_name, precision in zip(args.dataset.split(','), precision_list):
                print('  {}: {}'.format(dataset_name, precision))
                f.write('{}: {}\n'.format(dataset_name, precision))

    torch.distributed.barrier()
