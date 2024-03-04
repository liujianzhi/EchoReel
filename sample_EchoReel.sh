CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python main.py \
                            --config configs/video_inject.yaml \
                            --devices 10 \
                            --name EchoReel \
                            --savedir logs \
                            --batch_size 6 \
                            --mode test \
                            --resume PATH_TO_CKPT \
                            --train_dataset dataset/U91.json \
                            --test_dataset dataset/U10.json