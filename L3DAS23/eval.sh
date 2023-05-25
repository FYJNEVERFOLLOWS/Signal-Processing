
python_path=/Work21/2021/wanghonglong/sepenv/bin/python
processed_path=/Work21/2021/wanghonglong/datasets/L3DAS23_processed_100
id=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Work21/2021/wanghonglong/sepenv/lib

tag=mmub_officail_baseline
CUDA_VISIBLE_DEVICES=$id nohup $python_path -u evaluate_task1.py --architecture mmub \
                            --is_causal True \
                            --save_sounds_freq 100 \
                            --segment_length 76736 \
                            --predictors_path /CDShare3/L3DAS23/processed_train100_1mic/task1_predictors_test_uncut.pkl \
                            --target_path /CDShare3/L3DAS23/processed_train100_1mic/task1_target_test_uncut.pkl \
                            --model_path  RESULTS/$tag/checkpoint \
                            --results_path RESULTS/$tag/metrics \
                            > RESULTS/$tag/eval.log 2>&1 &