import os, sys 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tqdm import tqdm
import itertools
from PIL import Image
from infmllm.processors.processors import InstructImageProcessor
import torch

from infmllm.models import build_model


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


def collate_fn(batches):
    images = torch.stack([_[0] for _ in batches], dim=0)
    prompts = [_[1] for _ in batches]
    img_names = [_[2] for _ in batches]
    questions = [_[3] for _ in batches]
    gts = [_[4] for _ in batches]

    return {
        'image': images,
        'prompts': prompts,
        'img_names': img_names,
        'questions': questions,
        'gts': gts,
    }

class MmeDataset(torch.utils.data.Dataset):
    def __init__(self, prompt, lines, image_dir, vis_processor):
        self.prompt = prompt
        self.lines = lines
        self.image_dir = image_dir
        self.vis_processor = vis_processor
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        img, question, gt = line.strip().split('\t')

        img_path = os.path.join(self.image_dir, img)
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.image_dir, 'images', img)
        assert os.path.isfile(img_path), '{} not exist.'.format(img_path)
        image = Image.open(img_path).convert('RGB')

        image = self.vis_processor(image)

        return image, self.prompt.format(question), os.path.basename(img_path), question, gt


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
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument('--model_type', type=str, default="infmllm_inference_llama")
    # for vit model
    parser.add_argument('--vit_model', type=str, default="eva_clip_g")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--vit_lora_config', type=str, default="")
    # for vision adapter
    parser.add_argument('--vision_adapter', type=str, default="pooler")
    parser.add_argument('--pool_out_size', type=int, default=16)
    # for lm model
    parser.add_argument('--lm_model', type=str, default="pretrain_models/llama-2-7b-chat-hf/")
    parser.add_argument('--lm_tokenizer', type=str, default="pretrain_models/llama-2-7b-chat-hf/")
    parser.add_argument('--lm_lora_config', type=str, default="")
    parser.add_argument('--precision', type=str, default="amp_bf16")

    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--conv_version', type=str, default="vicuna_v1")
    parser.add_argument('--pdb_debug', action='store_true')

    args = parser.parse_args()
    if args.pdb_debug:
        import pdb; pdb.set_trace()
    
    # Do some check
    if 'vicuna' in args.conv_version:
        assert 'vicuna' in args.lm_model, 'conv_version and  lm_model should be consistent.'
    if 'llama' in args.conv_version:
        assert 'llama' in args.lm_model, 'conv_version and  lm_model should be consistent.'
    

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    #root_dir = 'evaluate/mme/eval_tool/Your_Results'
    root_dir = 'evaluate/mme/eval_tool/Your_Results_LLaVa_V1.5'
    out_dir = os.path.join(os.path.splitext(args.checkpoint)[0] + '_mme')
    os.makedirs(out_dir, exist_ok=True)

    image_processor = InstructImageProcessor(image_size=args.image_size)

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

    if args.conv_version == 'vicuna_v1':
        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        prompt = system + " " + "USER: <image><ImageHere>\n{} ASSISTANT: "
    else:
        raise ValueError()

    for filename in os.listdir(root_dir):
        image_dir = os.path.join("datasets/MME_Benchmark", os.path.basename(filename)[:-4])
        with open(os.path.join(root_dir, filename), 'r') as fin:
            lines = fin.read().splitlines()

            dataset = MmeDataset(prompt, lines, image_dir, image_processor)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=2,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )

            outputs = []
            for _, samples in tqdm(enumerate(dataloader)):
                samples['image'] = samples['image'].cuda()
                answers = model.predict_answers(
                    samples=samples,
                    num_beams=5,
                    max_len=100,
                    min_len=1,
                    length_penalty=0
                )
                for img_name, question, gt, pred in zip(samples['img_names'], samples['questions'], samples['gts'], answers):
                    pred = pred.replace('\n', ' ')
                    outputs.append('\t'.join([img_name, question, gt, pred]))

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            with open(os.path.join(out_dir, filename), 'w') as f:
                for d in merged_outputs:
                    f.write('{}\n'.format(d))
