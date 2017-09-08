## Marshall rule

### intro
We study, in this project, how neural networks capturing known phyiscal rules.

### usage

Run training (by default)
```
python main.py 
```

There are following options
* is_train
* cell_size
* num_layers
* task_name
* DATA_PATH
* LABEL_PATH
* NUM_EPOCH
* BATCH_SIZE

for example
```
python main.py --is_train=True --cell_size=16 --num_layers=2 --DATA_PATH=datasetMerged/states_J0.txt --LABEL_PATH=datasetMerged/sign_J0.txt --task_name=demo --SAVE_CKPT_PER_STEPS=100000
```

note: emprically, datasetConfig takes 200k iterations to converge, and J1J2 takes about 100k.

After
```
python main.py --is_train=False
```

### todo
* ipython notebook for testing visualization 
* trained model