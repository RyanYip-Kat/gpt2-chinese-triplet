import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder
from torch.utils.data import Dataset, DataLoader

def build_files(raw_data_path, tokenized_data_path, full_tokenizer, num_pieces):
    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = []
        #lines = json.load(f)
        #lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        for line in  f :
            line = line.rstrip().split("#")[1]
            lines.append(line + ' [SEP] ')

    single = ''.join(lines)
    len_single = len(single)
    if not os.path.exists(tokenized_data_path):
        os.makedirs(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        single_ids = full_tokenizer.convert_tokens_to_ids(
            full_tokenizer.tokenize(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)]))
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in single_ids[:-1]:
                f.write(str(id) + ' ')
            f.write(str(single_ids[-1]))
            f.write('\n')

    print('finish')


def load_sen2keyword(filename):
    texts = []
    labels = []
    for i,line in enumerate(open(filename,encoding="utf-8")):
        line = line.rstrip()
        label,text = line.split("#")
        texts.append(text)
        labels.append(label)
    return texts,labels

class sen2keyword(Dataset):
    def __init__(self,raw_data,full_tokenizer):
        super(sen2keyword,self).__init__()
        self.raw_data = raw_data
        self.full_tokenizer = full_tokenizer
        self.texts,self.labels = self.load_sen2keyword(self.raw_data)

    def load_sen2keyword(self,filename):
        texts = []
        labels = []
        for i,line in enumerate(open(filename,encoding="utf-8")):
            line = line.rstrip()
            label,text = line.split("#")
            texts.append(text)
            labels.append(label)
        return texts,labels


    def __len__(self):
        return len(self.texts)

    def __getitem__(self,index):
        text = self.texts[index]
        label = self.labels[index]
        #text_ids = self.full_tokenizer.convert_tokens_to_ids(self.full_tokenizer.tokenize(text))
        #label_ids = self.full_tokenizer.convert_tokens_to_ids(self.full_tokenizer.tokenize(label))
        sample ={}
        sample["text"] = text
        sample["label"] = label
        return sample

class Gpt2GenerateKeywordCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        labels = self.use_tokenizer(text=labels, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':labels})
        inputs.update({'texts':inputs})

        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data', default='sen2keyword.txt', type=str, required=False, help='原始训练语料')
    #parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
    #                    help='tokenized语料存放位置')
    #parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    #parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    #full_tokenizer["max_len"] = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data = args.raw_data
    #tokenized_data_path = args.tokenized_data_path
    #raw = args.raw  # 选择是否从零开始构建数据集
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    #num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("loading dataset...")
    dataset = sen2keyword(raw_data,full_tokenizer)
    dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader) * epochs
    print('total steps = {}'.format(total_steps))
    #if raw:
    #    print('building files')
    #    build_files(data_path=raw_data, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
    #                full_tokenizer=full_tokenizer, min_length=min_length)
    #    print('files built')

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    #full_len = 0
    #print('calculating total steps')
    #for i in tqdm(range(num_pieces)):
    #    with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
    #        full_len += len([int(item) for item in f.read().strip().split()])
    #total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    #scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
    #                                                      t_total=total_steps)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps = total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    #    multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        for step,(batch_inputs,label_inputs) in enumerate(dataloader):  
                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=label_inputs)
                loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {}  of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                    running_loss = 0
                overall_step += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + '/model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + '/model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + '/model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + '/final_model'):
        os.mkdir(output_dir + '/final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + '/final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    #main()
    from tokenizations import tokenization_bert
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file="gpt2-chinese-youlai/final_model/vocab.txt")
    raw_data = "sen2keyword.txt"
    dataset = sen2keyword(raw_data,full_tokenizer)
    gpt2_collator = Gpt2GenerateKeywordCollator(use_tokenizer=full_tokenizer,
                                                max_sequence_len=128)

    dataloader =  DataLoader(dataset, batch_size=8, shuffle=True,collate_fn=gpt2_collator)
    for sample in dataloader:
        batch_inputs = sample["texts"]
        label_inputs = sample["labels"]
        print(batch_inputs.shape)
        print(label_inputs.shape)
