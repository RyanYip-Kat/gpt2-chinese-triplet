python gpt2_finetune_classification.py --train /path/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt --valid /path/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/test.txt --pretrained_model  gpt2-chinese-youlai/final_model/ --vocab gpt2-chinese-youlai/final_model/vocab.txt --batch_size 32 --max_length 64  --epoch 5