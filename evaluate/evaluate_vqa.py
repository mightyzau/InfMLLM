import os, sys 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional
from PIL import Image

import torch
from tqdm import tqdm

from infmllm.processors.processors import Blip2ImageEvalProcessor, BlipQuestionProcessor
from evaluate.vqa_metric import VQA, VQAEval
from infmllm.models import build_model


ds_collections = {
    'vqav2_val': {
        'vis_root': 'datasets/vqav2/',
        'train': 'datasets/vqav2/vqav2_train.jsonl',
        'test': 'datasets/vqav2/vqav2_val.jsonl',
        'question': 'datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'datasets/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'vis_root': 'datasets/vqav2/',
        'train': 'datasets/vqav2/vqav2_train.jsonl',
        'test': 'datasets/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'vis_root': 'datasets/okvqa/',
        #'train': 'datasets/okvqa/okvqa_train.jsonl',
        'test': 'datasets/okvqa/okvqa_val.jsonl',
        'question': 'datasets/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'datasets/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'vis_root': "datasets/TextVQA",
        'train': 'datasets/TextVQA/textvqa_train.jsonl',
        'test': 'datasets/TextVQA/textvqa_val.jsonl',
        'question': 'datasets/TextVQA/textvqa_val_questions.json',
        'annotation': 'datasets/TextVQA/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'datasets/vizwiz/vizwiz_train.jsonl',
        'test': 'datasets/vizwiz/vizwiz_val.jsonl',
        'question': 'datasets/vizwiz/vizwiz_val_questions.json',
        'annotation': 'datasets/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'datasets/vizwiz/vizwiz_train.jsonl',
        'test': 'datasets/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'vis_root': 'datasets/DocVQA',
        'train': 'datasets/DocVQA/train.jsonl',
        'test': 'datasets/DocVQA/val.jsonl',
        'annotation': 'datasets/DocVQA/val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'vis_root': 'datasets/DocVQA',
        'train': 'datasets/DocVQA/train.jsonl',
        'test': 'datasets/DocVQA/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'vis_root': 'datasets/ChartQA/',
        'train': 'datasets/ChartQA/train_human.jsonl',
        'test': 'datasets/ChartQA/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'vis_root': 'datasets/ChartQA/',
        'train': 'datasets/ChartQA/train_augmented.jsonl',
        'test': 'datasets/ChartQA/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'vis_root': 'datasets/gqa/images',
        'test': 'datasets/gqa/annotations/converted/testdev_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'vis_root': 'datasets/ocr-vqa',
        'train': 'datasets/ocr-vqa/ocrvqa_train.jsonl',
        'test': 'datasets/ocr-vqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'vis_root': 'datasets/ocr-vqa',
        'train': 'datasets/ocr-vqa/ocrvqa_train.jsonl',
        'test': 'datasets/ocr-vqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'vis_root': 'datasets/AI2Diagram/ai2d',
        'train': 'datasets/AI2Diagram/ai2d/train.jsonl',
        'test': 'datasets/AI2Diagram/ai2d/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches):
    images = torch.stack([_['image'] for _ in batches], dim=0)
    prompts = [_['prompt'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    answers = [_['answer'] for _ in batches]

    return {
        'image': images,
        'prompts': prompts,
        'question_ids': question_ids,
        'answers': answers
    }


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, vis_root, test, prompt, vis_processor, text_processor):
        self.vis_root = vis_root
        self.test = open(test).readlines()
        self.prompt = prompt
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, answer = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        image = os.path.join(self.vis_root, image)
        
        image = Image.open(image).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(question)
        prompt = self.prompt.format(question)
        
        return {
            'image': image,
            'prompt': prompt,
            'question_id': question_id,
            'answer': answer
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

    parser.add_argument('--length_penalty', type=float, default=0)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--prompt', type=str)

    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pdb_debug', action='store_true')
    args = parser.parse_args()
    args.apply_lemmatizer = eval(args.apply_lemmatizer)

    if args.pdb_debug:
        import pdb; pdb.set_trace()

    if args.pdb_debug:
        pass 
    else:
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

    random.seed(args.seed)

    all_results = {}
    for dataset_name in args.dataset.split(','):
        dataset_name = dataset_name.strip()

        results_file = os.path.splitext(args.checkpoint)[0] + f'_{dataset_name}_{args.image_size}.json'
        if os.path.isfile(results_file):
            print('result file aleady exist, ignore.')
            merged_outputs = json.load(open(results_file))
        else:

            image_processor = Blip2ImageEvalProcessor(image_size=args.image_size)
            text_processor = BlipQuestionProcessor()
            dataset = VQADataset(
                vis_root=ds_collections[dataset_name]['vis_root'],
                test=ds_collections[dataset_name]['test'],
                prompt=args.prompt,
                vis_processor=image_processor,
                text_processor=text_processor
            )
            if args.pdb_debug:
                sampler = None 
            else:
                sampler = InferenceSampler(len(dataset))
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )

            outputs = []
            for _, samples in tqdm(enumerate(dataloader)):
                samples['image'] = samples['image'].cuda()
                if dataset_name in ['okvqa_val']:
                    samples['apply_lemmatizer'] = True

                pred_answers = model.predict_answers(
                    samples=samples,
                    num_beams=args.num_beams,
                    max_len=ds_collections[dataset_name]['max_new_tokens'],
                    min_len=args.min_len,
                    length_penalty=args.length_penalty
                )
                question_ids = samples['question_ids']
                gt_answers = samples['answers']

                for question_id, answer, annotation in zip(question_ids, pred_answers, gt_answers):
                    if dataset_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val', 'vizwiz_val']:
                        outputs.append({
                            'question_id': question_id,
                            'answer': answer,
                        })
                    elif dataset_name in ['docvqa_val', 'infographicsvqa', 'gqa_testdev', 'ocrvqa_val', 'ocrvqa_test']:
                        outputs.append({
                            'questionId': question_id,
                            'answer': answer,
                            'annotation': annotation,
                        })
                    elif dataset_name in ['ai2diagram_test']:
                        outputs.append({
                            'image': question_id,
                            'answer': answer,
                            'annotation': annotation,
                        })
                    elif dataset_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                        outputs.append({
                            'answer': answer,
                            'annotation': annotation,
                        })
                    elif dataset_name in ['docvqa_test']:
                        outputs.append({
                            'questionId': question_id,
                            'answer': answer,
                        })
                    elif dataset_name in ['vizwiz_test']:
                        outputs.append({
                            'image': question_id,
                            'answer': answer,
                        })
                    else:
                        raise NotImplementedError

            torch.distributed.barrier()

            world_size = torch.distributed.get_world_size()
            merged_outputs = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

            merged_outputs = [json.loads(_) for _ in merged_outputs]
            merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")
            #time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            #results_file = f'{dataset_name}_{time_prefix}_s{args.seed}.json'
            if not os.path.isfile(results_file):
                json.dump(merged_outputs, open(results_file, 'w'), ensure_ascii=False)

            if ds_collections[dataset_name]['metric'] == 'vqa_score':
                vqa = VQA(ds_collections[dataset_name]['annotation'],
                        ds_collections[dataset_name]['question'])
                results = vqa.loadRes(
                    resFile=results_file,
                    quesFile=ds_collections[dataset_name]['question'])
                vqa_scorer = VQAEval(vqa, results, n=2)
                vqa_scorer.evaluate()

                print(vqa_scorer.accuracy)
                all_results[dataset_name] = vqa_scorer.accuracy

            elif ds_collections[dataset_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                        open(results_file, 'w'),
                        ensure_ascii=False)
                print('python evaluate/infographicsvqa_eval.py -g ' +
                    ds_collections[dataset_name]['annotation'] + ' -s ' +
                    results_file)
                os.system('python evaluate/infographicsvqa_eval.py -g ' +
                        ds_collections[dataset_name]['annotation'] + ' -s ' +
                        results_file)
            elif ds_collections[dataset_name]['metric'] == 'relaxed_accuracy':
                r = evaluate_relaxed_accuracy(merged_outputs)
                print({'relaxed_accuracy': r})
                all_results[dataset_name] = r
            elif ds_collections[dataset_name]['metric'] == 'accuracy':
                if 'gqa' in dataset_name:
                    for entry in merged_outputs:
                        response = entry['answer']
                        response = response.strip().split('.')[0].split(
                            ',')[0].split('!')[0].lower()
                        if 'is ' in response:
                            response = response.split('is ')[1]
                        if 'are ' in response:
                            response = response.split('are ')[1]
                        if 'a ' in response:
                            response = response.split('a ')[1]
                        if 'an ' in response:
                            response = response.split('an ')[1]
                        if 'the ' in response:
                            response = response.split('the ')[1]
                        if ' of' in response:
                            response = response.split(' of')[0]
                        response = response.strip()
                        entry['answer'] = response
                r = evaluate_exact_match_accuracy(merged_outputs)
                print({'accuracy': r})
                all_results[dataset_name] = r

        torch.distributed.barrier()
    

    if torch.distributed.get_rank() == 0:
        print()
        print('*' * 50)        
        file = os.path.splitext(args.checkpoint)[0] + '_eval.txt'
        with open(file, 'a') as f:
            for dataset_name in all_results:
                print('  {}: {}'.format(dataset_name, all_results[dataset_name]))
                f.write('{}: {}\n'.format(dataset_name, all_results[dataset_name]))
