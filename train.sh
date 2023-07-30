
DATASET='plant-village-healthy-to-plant-village-rust'
EXPERIMENT='test'
VAL_FREQ=10
PRINT_FREQ=5
SAVE_LATEST_FREQ=1000
SAVE_EPOCH_FREQ=100

python train.py --dataroot ./data/$DATASET \
                --name $EXPERIMENT \
                --CUT_mode CUT \
                --display_id 0 \
                --val_freq $VAL_FREQ \
                --print_freq $PRINT_FREQ \
                --save_latest_freq $SAVE_LATEST_FREQ \
                --save_by_iter \
                --save_epoch_freq $SAVE_EPOCH_FREQ