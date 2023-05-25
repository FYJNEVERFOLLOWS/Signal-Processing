#$ -N eab_train100_1mic_ljjcmse
#$ -cwd
#$ -e /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -o /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -l h=gpu09
echo "job start time: `date`"
echo "Which node:`hostname`"

tag=eabself_train360_2mic_cmse

python_path=/Work21/2021/wanghonglong/sepenv/bin/python
processed_path=/Work21/2021/wanghonglong/datasets/L3DAS23_sepprocessed_360_2mic
id=0,1,2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Work21/2021/wanghonglong/sepenv/lib


CUDA_VISIBLE_DEVICES=$id nohup $python_path -u train_sep_task1.py --path_csv_images_train /CDShare3/L3DAS23_Task1/L3DAS23_Task1_train_100/audio_image.csv \
                        --path_csv_images_test /CDShare3/L3DAS23_Task1/audio_image.csv \
                        --batch_size 6 \
                        --epochs 200 \
                        --architecture eab \
                        --loss cmse \
                        --training_path $processed_path/train/task1_train_path.pkl \
                        --validation_path $processed_path/validation/task1_validation_path.pkl \
                        --test_path $processed_path/test/task1_test_path.pkl \
                        --results_path RESULTS/$tag \
                        --checkpoint_dir RESULTS/$tag \
                        --load_model /Work21/2021/wanghonglong/codeWork/L3DAS23/RESULTS/eabself_train360_2mic_cmse/10_checkpoint \
                        > logs/${tag}.log 2>&1 &
