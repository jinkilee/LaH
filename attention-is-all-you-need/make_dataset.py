import sys
sys.path.insert(0, '../')

from feature import write_to_one_text, remove_bos_eos
from dataset import load_dataset

sent_pairs = load_dataset(path='/heavy_data/jkfirst/workspace/git/LaH/dataset/')
sent_pairs = list(map(lambda x: remove_bos_eos(x), sent_pairs))

write_to_one_text(sent_pairs)
