import re

def remove_bos_eos(pair):
	src, trg = pair
	src = re.sub('<start> ', '', src)
	src = re.sub(' <end>', '', src)
	trg = re.sub('<start> ', '', trg)
	trg = re.sub(' <end>', '', trg)
	return [src, trg]

def write_to_one_text(sent_pairs):
	with open('./spm_src.txt', 'w', encoding='utf8') as src_f:
		for sp in sent_pairs:
			src_f.write(sp[0] + '\n')

	with open('./spm_trg.txt', 'w', encoding='utf8') as trg_f:
		for sp in sent_pairs:
			trg_f.write(sp[1] + '\n')

