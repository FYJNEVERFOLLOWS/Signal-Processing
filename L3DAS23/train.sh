#$ -N eab_train100_2mic_cmse
#$ -cwd
#$ -e /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -o /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -l h=gpu06
echo "job start time: `date`"
echo "Which node:`hostname`"

tag=_mmub_train100_2mic_l1

python_path=/Work21/2021/wanghonglong/sepenv/bin/python
processed_path=/Work21/2021/wanghonglong/datasets/L3DAS23_processed_100
id=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Work21/2021/wanghonglong/sepenv/lib


CUDA_VISIBLE_DEVICES=$id $python_path -u train_baselib_task1.py --path_csv_images_train /CDShare3/L3DAS23_Task1/L3DAS23_Task1_train_100/audio_image.csv \
                        --path_csv_images_test /CDShare3/L3DAS23_Task1/audio_image.csv \
                        --batch_size 16 \
                        --epochs 200 \
                        --architecture 'MIMO_UNet_Beamforming' \
                        --loss 'L1' \
                        --training_predictors_path $processed_path/task1_predictors_train.pkl \
                        --training_target_path $processed_path/task1_target_train.pkl \
                        --validation_predictors_path $processed_path/task1_predictors_validation.pkl \
                        --validation_target_path $processed_path/task1_target_validation.pkl \
                        --test_predictors_path $processed_path/task1_predictors_test.pkl \
                        --test_target_path $processed_path/task1_target_test.pkl \
                        --results_path RESULTS/$tag \
                        --checkpoint_dir RESULTS/$tag \
                        > logs/train_${tag}.log 2>&1 &
