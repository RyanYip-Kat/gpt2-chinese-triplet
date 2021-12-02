python train.py --model_checkpoint /path/huggingface-models/CDial-GPT_LCCC-large/ --pretrained --data_path /path/sentence_triplet_CDial-GPT.json  --num_workers 1 --scheduler linear --device cuda --gpt2
python infer.py --gpt2 --datapath /path/sentence_triplet_CDial-GPT_test.json --out_path sentence_triplet_test.txt --model_checkpoint  runs/Nov29_07-06-52_98cad14aa202/ --device cuda
