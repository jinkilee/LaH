
templates= '--input={} --pad_id={} --bos_id={} --eos_id={} --unk_id={} --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}'

pad_id = 0  #<pad> token을 0으로 설정
bos_id = 1 #<start> token을 1으로 설정
eos_id = 2 #<end> token을 2으로 설정
unk_id = 3 #<unknown> token을 3으로 설정

src_vocab_size = 5000 # vocab 사이즈
trg_vocab_size = 3000 # vocab 사이즈

src_prefix = 'spm-src-{}'.format(src_vocab_size) # 저장될 tokenizer 모델에 붙는 이름
trg_prefix = 'spm-trg-{}'.format(trg_vocab_size) # 저장될 tokenizer 모델에 붙는 이름

src_input_file = "./spm_src.txt"
trg_input_file = "./spm_trg.txt"

character_coverage = 1.0 # to reduce character set 
model_type ='unigram' # Choose from unigram (default), bpe, char, or word

MIN_FREQ = 2
