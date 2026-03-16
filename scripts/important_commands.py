
# 1- build the codebook  (gt images ----->net_g_latest.pth)
    # python basicsr/train.py -opt options/CodeFormer_stage2.yml
    
# 2- generate latent code [gt images  -------> ('./experiments/my_identity_codes.pth') ]
    ##python scripts/generate_latent_gt.py   --test_path /data1/hs_denoising/codeformer_dataset/codeformer_val/gt_overfit   --ckpt_path ./experiments/20260310_001306_VQGAN_Stage1_Personalized_50/models/net_g_latest.pth   --codebook_size 1024   --save_root ./experiments/my_identity_codes

# 3- run stage 2 [my_identity_codes.pth + net_g_latest.pth + gt|input -------------> ?]
    #  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.run --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch > output_stage_2.log 2>&1 &
    

# 3- run stage 3 [my_identity_codes.pth + net_g_latest.pth + gt|input -------------> ?]
    #  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.run --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher pytorch > output_stage_3.log 2>&1 &    
# before command
# set PYTHONPATH=.
# python scripts/crop_align_face.py -i E:\Code_store\CodeFormer\datasets\sample_finetune_data\lq -o E:\Code_store\CodeFormer\datasets\sample_finetune_data\lq_prepared
# python basicsr/train.py -opt options/CodeFormer_stage2.yml
