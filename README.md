
### Set up
```
- python==3.6.7
- torch==1.7.0
- torchvision==0.8.0

# other python/pytorch version might also work
```

### Data preparation

Download Nell dataset from [github](https://github.com/alexhw15/HiRe/releases/download/ckpt/Nell-data-Hire.zip) and Wiki dataset from [github](https://github.com/alexhw15/HiRe/releases/download/ckpt/Wiki-data-Hire.zip).

Download pre-trained embedding 

#### Train and test script examples:

To train HiRe on Nell-One under 1-shot setting:

```
python main.py --dataset NELL-One --few 1 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step train
```

To test HiRe on Wiki-One under 3-shot setting:
```
python main.py --dataset Wiki-One --data_path ./Wiki-Hire/ --few 3 --prefix example-test-wiki --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step test
```

To test HiRe on Nell-One under 5-shot setting using checkpoints:

```
python main.py --dataset NELL-One --few 5 --prefix example-test-ckpt-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --eval_ckpt ./best_ckpt/nell_5shot_best.ckpt --step test
```

