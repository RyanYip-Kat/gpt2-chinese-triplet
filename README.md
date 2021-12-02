# gpt2-chinese-triplet
##### GPT2 中文版本的应用
##### 参考
* https://github.com/Morizeyao/GPT2-Chinese

##### 使用
* [finetune gpt2 model](./finetune_gpt2_chinese)
   主要是使用自己的数据finetune gpt2模型，使得模型学习自己的文本知识
   ```python
   python finetune_gpt2_chinese/train_single.py --***
   ```
   
* [训练分类器](./gpt2_chinese_classification)
  训练文本分类器，格式类似（sentence y1);具体格式可以参考[datatype](./gpt2_chinese_classification/intention_cls_data_uniq_train.txt)
   ```python
   python gpt2_chinese_classification/gpt2_finetune_classification.py --***
   ```
   
* [句子到三元组](./CDial-GPT-triplete-classifier)
  [数据格式](./CDial-GPT-triplete-classifier/sentence_triplet_CDial-GPT_test.json)
   
  [代码使用方法](./CDial-GPT-triplete-classifier/run.sh)
   ```python
   python CDial-GPT-triplete-classifier/train.py
   ```
  
 * [句子+意图 ->如果...那么  范式](./CDial-GPT-IntentionAndText)
   训练一个*句子+意图* 到 *如果...那么...* 的固定句式，参考[run.sh](./CDial-GPT-IntentionAndText/run.sh)；[数据格式](./CDial-GPT-IntentionAndText/intention_text_CDial-GPT.json)
   ```python
   python CDial-GPT-IntentionAndText/train.py
   python CDial-GPT-IntentionAndText/infer.py
   ...
   ```
   
在训练这些应用模型时候，先将gpt2用自己的大量相关文本finetune，再使用finetune之后的的model 作为 具体模型的checkpoint
