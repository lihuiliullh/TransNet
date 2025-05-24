nohup python main_pretrain.py --dataset NELL-One --few 5 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 3 --step train > output_nell_5_new.txt 2>&1 & 

nohup python main_pretrain.py --dataset NELL-One --few 1 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 3 --step train > output_nell_1_new.txt 2>&1 & 


nohup python main.py --dataset Wiki-One --data_path Wiki-HiRe --few 5 --prefix example-train-wiki --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 3 --step train > output_wiki_5_new.txt 2>&1 & 



### For NELL dataset, directly run main_pretrain.py

For each knowledge graph, their are pretrained embeddings, which stored in Nell-HiRe and Wiki-HiRe respectively. 
other data, such as few shot or candidate data are also stored inside. 


nohup python main_hire.py --dataset NELL-One --few 1 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 3 --step train > output_nell_1_hire.txt 2>&1 & 
