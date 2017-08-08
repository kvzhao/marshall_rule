
# data information
DATA_PATH = "datasetConfig/states.txt"
LABEL_PATH = "datasetConfig/sign.txt"
DATA_NORMALIZED = False

MODEL = 'RNN'

# ~ 2000 steps per epoch
NUM_EPOCH = 2000
BATCH_SIZE = 32

# batch size of testing data
TEST_BATCH_SIZE = 1024

# records summaries per steps
EVAL_PER_STEPS = 500
