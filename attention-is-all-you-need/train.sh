#!/bin/bash
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--master_addr="127.0.0.1" \
	--master_port=10002 \
	train.py \
		--epoch=1000 \
		--opt-level=O1
