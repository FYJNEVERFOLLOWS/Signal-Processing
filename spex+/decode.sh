#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N test_baseline_uttsisdr

# resource requesting, e.g. for gpu use
#$ -l h=gpu06

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh.
echo `hostname`

gpuid=0
echo "gpuid: ${gpuid}"

/Work18/2020/lijunjie/anaconda3/envs/torch1.8/bin/python3.8 \
./nnet/separate.py --cpt_dir "/Work21/2021/fuyanjie/pycode/libri_seg_spex+/exp/exp0109_uttsisdr" \
--gpuid $gpuid \
--separated_dir "0109-uttsisdr"


sleep 10
echo "job end time:`date`"
