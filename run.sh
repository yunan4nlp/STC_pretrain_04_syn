export CUDA_VISIBLE_DEVICES=1
export LC_CTYPE=en_US.UTF-8

nohup /home/zrr/anaconda3/bin/python3.6 -u driver/Train.py --tgt_word_file tgt_vocab.txt --thread 1 --use-cuda --config en-vi-transformer.cfg.model1  > log 2>&1 &
