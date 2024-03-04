CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py \
                            --config configs/video_inject.yaml \
                            --devices 4 \
                            --name EchoReel \
                            --savedir logs \
                            --batch_size 4 \
                            --train_dataset dataset/U91.json \
                            --test_dataset dataset/U10.json