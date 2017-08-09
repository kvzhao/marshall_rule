
# data information
DATA_PATH = "datasetConfig/states.txt"
LABEL_PATH = "datasetConfig/sign.txt"
DATA_NORMALIZED = False

MODEL = 'RNN'

# ~ 2000 steps per epoch
NUM_EPOCH = 200
BATCH_SIZE = 32

# batch size of testing data
TEST_BATCH_SIZE = 1024

# records summaries per steps
EVAL_PER_STEPS = 500

SAVE_CKPT_PER_STEPS = 50000
