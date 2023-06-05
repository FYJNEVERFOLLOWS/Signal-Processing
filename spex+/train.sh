#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N libri_seg_segsdr

# resource requesting, e.g. for gpu use
#$ -l h=gpu07

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#chmod a+x run.sh.
echo `hostname`

gpuid=2
cpt_dir=exp/exp0109_segsdr
batch_size=3
epochs=50
echo "gpuid: ${gpuid}"

/Work18/2020/lijunjie/anaconda3/envs/torch1.8/bin/python3.8 ./nnet/train.py --gpu $gpuid --epochs $epochs --batch-size $batch_size --checkpoint $cpt_dir

sleep 10
echo "job end time:`date`"
