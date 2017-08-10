## Marshall rule

### intro
We study, in this project, how neural networks capturing known phyiscal rules.

### usage

Run training 
```
python main.py --is_train=True 
```

There are following options
* is_train
* task_name
* DATA_PATH
* LABEL_PATH
* NUM_EPOCH
* BATCH_SIZE

note: emprically, datasetConfig takes 20k iteration to converge, and J1J2 takes about 10k.

For testing
```
python main.py --is_train=False
```

### todo
* ipython notebook for testing visualization 
* trained model
* dynamic rnn