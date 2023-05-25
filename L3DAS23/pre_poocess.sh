#$ -N process_both_2mic_5sec
#$ -cwd
#$ -e /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -o /Work21/2021/wanghonglong/codeWork/L3DAS23/logs
#$ -l h=gpu09
echo "job start time: `date`"
echo "Which node:`hostname`"

python_path=/Work21/2021/wanghonglong/sepenv/bin/python
TASK1_DataPATH=/CDShare3/L3DAS23/Task1
OUT_PATH=/Work21/2021/wanghonglong/datasets/L3DAS23_processed_100_2mic

$python_path preprocessing.py --task 1 \
                --input_path $TASK1_DataPATH \
                --output_path $OUT_PATH \
                --training_set train100\
                --pad_length 5 \
                --num_mics 2