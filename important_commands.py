
# run stage 1 
# export PYTHONPATH=$PYTHONPATH:. python basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher none


# run stage 2 
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3
python basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher none


export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch


fuser -v /dev/nvidia1


tensorboard --logdir experiments/20260227_082302_CodeFormer_stage2 --port 6006


# run stage 3 
# export PYTHONPATH=$PYTHONPATH:.
# python basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher none
