import argparse
import os
import pickle
import random
import sys
# import Image,

import librosa
import numpy as np

import utility_functions as uf

"""
To do
both代码没有写
path不写成列表可以储存为pickle吗
在process_folder中是否可以划分好train/val可能差一个

"""


'''
Process the unzipped dataset folders and output numpy matrices (.pkl files)
containing the pre-processed data for task1 and task2, separately.
Separate training, validation and test matrices are saved.
Command line inputs define which task to process and its parameters.
The preprocessing of L3DAS23 was designed to be exploited in both audiovisual 
AND audio-only track. Teams participating only in audio-only may find the 
preprocessing of L3DAS22, available in the relevant GitHub repo, useful.
'''

sound_classes_dict_task2 = {'Chink_and_clink':0,
                           'Computer_keyboard':1,
                           'Cupboard_open_or_close':2,
                           'Drawer_open_or_close':3,
                           'Female_speech_and_woman_speaking':4,
                           'Finger_snapping':5,
                           'Keys_jangling':6,
                           'Knock':7,
                           'Laughter':8,
                           'Male_speech_and_man_speaking':9,
                           'Printer':10,
                           'Scissors':11,
                           'Telephone':12,
                           'Writing':13}


def preprocessing_task1(args):
    '''
    predictors output: a list of two elements, where the first contains
                            > ambisonics mixture waveforms
                              Matrix shape: -x: data points
                                            -4 or 8: ambisonics channels
                                            -signal samples
                       and the second element contains
                            > a list of image filenames, 
                              each corresponding to the respective audio

    target output: monoaural clean speech waveforms
                   Matrix shape: -x: data points
                                 -1: it's monoaural
                                 -signal samples
    '''
    sr_task1 = 16000
    max_file_length_task1 = 12

    def pad(x, size):
        #pad all sounds to 4.792 seconds to meet the needs of Task1 baseline model MMUB
        length = x.shape[-1]
        if length > size:
            pad = x[:,:size]
        else:
            pad = np.zeros((x.shape[0], size))
            pad[:,:length] = x
        return pad

    def process_folder(folder, args, category, split_point):
        #process single dataset folder
        print('Processing ' + folder + ' folder...')
        # predictors = []
        # target = []
        predictors_pts = []
        count = 0
        main_folder = os.path.join(args.input_path, folder) #由于原始数据文件夹重复，还是改代码这里吧
        # with open(os.path.join(main_folder, "audio_image.csv"))
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)
        
        if category == 'train':
            train_out_dir = os.path.join(args.output_path, 'train')
            if not os.path.isdir(train_out_dir):
                os.makedirs(train_out_dir)
            val_out_dir = os.path.join(args.output_path, 'validation')
            if not os.path.isdir(val_out_dir):
                os.makedirs(val_out_dir)
        elif category == 'test':
            test_out_dir = os.path.join(args.output_path, 'test')
            if not os.path.isdir(test_out_dir):
                os.makedirs(test_out_dir)
        elif category == 'uncut':
            uncut_out_dir = os.path.join(args.output_path, 'test_uncut')
            if not os.path.isdir(uncut_out_dir):
                os.makedirs(uncut_out_dir)

        

        data_path = os.path.join(main_folder, 'data')

        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']  #filter files with mic B
        for sound in data:
            # print(count)
            sound_path = os.path.join(data_path, sound)
            target_path = '/'.join((sound_path.split('/')[:-2] + ['labels'] + [sound_path.split('/')[-1]]))  #change data with labels
            target_path = target_path[:-6] + target_path[-4:]  #remove mic ID
            #target_path = sound_path.replace('data', 'labels').replace('_A', '')  #old wrong line
            samples, sr = librosa.load(sound_path, sr=sr_task1, mono=False)
            # image = 
            #samples = pad(samples)
            if args.num_mics == 2:  # if both ambisonics mics are wanted
                #stack the additional 4 channels to get a (8, samples) shap
                B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                samples_B, sr = librosa.load(B_sound_path, sr=sr_task1, mono=False)
                samples = np.concatenate((samples,samples_B), axis=-2)

            samples_target, sr = librosa.load(target_path, sr=sr_task1, mono=False)
            samples_target = samples_target.reshape((1, samples_target.shape[0]))
            print("count:{}, split_point:{}".format(count, split_point))
            #append to final arrays
            if args.segmentation_len is not None:
                #segment longer file to shorter frames
                #not padding if segmenting to avoid silence frames
                segmentation_len_samps = int(sr_task1 * args.segmentation_len)
                predictors_cuts, target_cuts = uf.segment_waveforms(samples, samples_target, segmentation_len_samps)
                for i in range(len(predictors_cuts)):
                    predictors_path = sound[:-6]
                    predictors_pts.append(predictors_path)
                    predictors = predictors_cuts[i]
                    target = target_cuts[i]

                    predictors_withpt = [predictors, predictors_path, target] #predictors_withpath_target
                    if count < split_point and category == 'train':
                        with open(os.path.join(train_out_dir, sound[:-6]+ '.pkl'), 'wb') as f:
                            pickle.dump(predictors_withpt, f, protocol=4)
                    elif count >= split_point and category == 'train':
                        with open(os.path.join(val_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                            pickle.dump(predictors_withpt, f, protocol=4)
                    elif category == 'test':
                        with open(os.path.join(test_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                            pickle.dump(predictors_withpt, f, protocol=4)
                    elif category == 'uncut':
                        with open(os.path.join(uncut_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                            pickle.dump(predictors_withpt, f, protocol=4)
            else:
                samples = pad(samples, size=int(sr_task1*args.pad_length))
                samples_target = pad(samples_target, size=int(sr_task1*args.pad_length))
                predictors_path = sound[:-6]
                predictors_pts.append(predictors_path)
                predictors = samples
                target = samples_target
                # target.append()

                predictors_withpt = [predictors, predictors_path, target] #predictors_withpath_target
                if count < split_point and category == 'train':
                    with open(os.path.join(train_out_dir, sound[:-6]+ '.pkl'), 'wb') as f:
                        pickle.dump(predictors_withpt, f, protocol=4)
                elif count >= split_point and category == 'train':
                    with open(os.path.join(val_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                        pickle.dump(predictors_withpt, f, protocol=4)
                elif category == 'test': 
                        with open(os.path.join(test_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                            pickle.dump(predictors_withpt, f, protocol=4)
                elif category == 'uncut': 
                    with open(os.path.join(uncut_out_dir, sound[:-6]+'.pkl'), 'wb') as f:
                        pickle.dump(predictors_withpt, f, protocol=4)
            # print("here!!!! ", samples.shape)
            count += 1
            if args.num_data is not None and count >= args.num_data:
                break

        return predictors_pts

    #process all required folders
    test_split_point = len(os.listdir(os.path.join(args.input_path, 'L3DAS23_Task1_dev', 'data'))) // 2  
    print("test split_point:{}".format(test_split_point))
    test_path= process_folder('L3DAS23_Task1_dev', args, "test", test_split_point)

    #split train set into train and development
    # split_point = int(len(predictors_train) * args.train_val_split)
    if args.training_set == 'train100':  #数据集里有下划线，所以这里加上。
        split_point = int(len(os.listdir(os.path.join(args.input_path, 'L3DAS23_Task1_train100', 'data'))) // 2 * args.train_val_split)
        print("train100 split_point:{}".format(split_point))
        predictors_path_train= process_folder('L3DAS23_Task1_train100', args, "train", split_point)
        train_path = predictors_path_train[:split_point]   #attention: changed training names
        validation_path = predictors_path_train[split_point:]
    elif args.training_set == 'train360':
        split_point = int(len(os.listdir(os.path.join(args.input_path, 'L3DAS23_Task1_train360', 'data'))) // 2 * args.train_val_split)
        print("train360 split_point:{}".format(split_point))
        predictors_path_train = process_folder('L3DAS23_Task1_train360', args, "train", split_point)
        train_path = predictors_path_train[:split_point]   #attention: changed training names
        validation_path = predictors_path_train[split_point:]
    elif args.training_set == 'both':
        len_100 = len(os.listdir(os.path.join(args.input_path, 'L3DAS23_Task1_train100'))) // 2
        len_360 = len(os.listdir(os.path.join(args.input_path, 'L3DAS23_Task1_train360'))) // 2
        #7727 + 29671条，train100全用来做训练集
        split_point = int((len_100 + len_360) * args.train_val_split) - len_100
        print("both split_point:{}".format(split_point))
        predictors_train_path100 = process_folder('L3DAS23_Task1_train100', args, "train", len_100)
        predictors_train_path360 = process_folder('L3DAS23_Task1_train360', args, "train", split_point)
        train_path = predictors_train_path100 + predictors_train_path360[:split_point]
        validation_path = predictors_train_path360[split_point:]
    
    # target_training = target_train[:split_point]
    
    # target_validation = target_train[split_point:]
    # predictors_test = predictors_path_test

    #save numpy matrices in pickle files
    print('Saving files')
    # if not os.path.isdir(args.output_path):
    #     os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, 'train','task1_train_path.pkl'), 'wb') as f:
        pickle.dump(train_path, f, protocol=4)
    with open(os.path.join(args.output_path, 'validation','task1_validation_path.pkl'), 'wb') as f:
        pickle.dump(validation_path, f, protocol=4)
    with open(os.path.join(args.output_path,'test', 'task1_test_path.pkl'), 'wb') as f:
        pickle.dump(test_path, f, protocol=4)
    # with open(os.path.join(args.output_path,'task1_target_train.pkl'), 'wb') as f:
    #     pickle.dump(target_training, f, protocol=4)
    # with open(os.path.join(args.output_path,'task1_target_validation.pkl'), 'wb') as f:
    #     pickle.dump(target_validation, f, protocol=4)
    # with open(os.path.join(args.output_path,'task1_target_test.pkl'), 'wb') as f:
    #     pickle.dump(target_test, f, protocol=4)

    #generate also a test set matrix with full-length samples, just for the evaluation
    print('processing uncut test set')
    args.pad_length = max_file_length_task1
    process_folder('L3DAS23_Task1_dev', args, 'uncut', test_split_point)


    print('Audio Matrices successfully saved')
    print('Training set shape: ', np.array(train_path).shape)
    print('Validation set shape: ', np.array(validation_path).shape)
    print('Test set shape: ', np.array(test_path).shape)


def preprocessing_task2(args):
    '''
    predictors output: a list of two elements, where the first contains
                            > ambisonics stft
                              Matrix shape: -x data points
                                            - num freqency bins
                                            - num time frames
                       and the second element contains
                            > a list of image filenames, 
                              each corresponding to the respective audio 
                        
    target output: matrix containing all active sounds and their position at each
                   100msec frame.
                   Matrix shape: -x data points
                                 -300: frames
                                 -168: 14 (classes) * 3 (max simultaneous sounds per frame)
                                       concatenated to 14 (classes) * 3 (max simultaneous sounds per frame) * 3 (xyz coordinates)
    '''
    sr_task2 = 32000
    sound_classes=['Chink_and_clink','Computer_keyboard','Cupboard_open_or_close',
             'Drawer_open_or_close','Female_speech_and_woman_speaking',
             'Finger_snapping','Keys_jangling','Knock',
             'Laughter','Male_speech_and_man_speaking',
             'Printer','Scissors','Telephone','Writing']
    file_size=30.0
    max_label_value = 360.  #maximum rho,theta,z value (serves for normalization)

    def process_folder(folder, args):
        print ('Processing ' + folder + ' folder...')
        predictors = []
        target = []
        predictors_path = []
        image_predictors = []
        data_path = os.path.join(folder, 'data')
        labels_path = os.path.join(folder, 'labels')

        data = os.listdir(data_path)
        data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']
        count = 0
        for sound in data:
            ov_set = sound.split('_')[-3]
            if ov_set in args.ov_subsets:  #if data point is in the desired subsets ov
                target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
                sound_path = os.path.join(data_path, sound)
                target_path = os.path.join(data_path, target_name)
                target_path = '/'.join((target_path.split('/')[:-2] + ['labels'] + [target_path.split('/')[-1]]))  #change data with labels
                #target_path = target_path.replace('data', 'labels')  #old
                samples, sr = librosa.load(sound_path, sr=sr_task2, mono=False)
                if args.num_mics == 2:  # if both ambisonics mics are wanted
                    #stack the additional 4 channels to get a (8, samples) shape
                    B_sound_path = sound_path[:-5] + 'B' +  sound_path[-4:]  #change A with B
                    #B_sound_path = sound_path.replace('A', 'B')  old
                    samples_B, sr = librosa.load(B_sound_path, sr=sr_task2, mono=False)
                    samples = np.concatenate((samples,samples_B), axis=-2)

                #compute stft
                stft = uf.spectrum_fast(samples, nperseg=args.stft_nperseg,
                                        noverlap=args.stft_noverlap,
                                        window=args.stft_window,
                                        output_phase=args.output_phase)

                #compute matrix label
                label = uf.csv_to_matrix_task2(target_path, sound_classes_dict_task2,
                                               dur=int(file_size), step=args.frame_len/1000., max_loc_value=max_label_value,
                                               no_overlaps=args.no_overlaps)  #eric func

                #segment into shorter frames
                if args.predictors_len_segment is not None and args.target_len_segment is not None:
                    #segment longer file to shorter frames
                    #not padding if segmenting to avoid silence frames
                    predictors_cuts, target_cuts = uf.segment_task2(stft, label, predictors_len_segment=args.predictors_len_segment,
                                                    target_len_segment=args.target_len_segment, overlap=args.segment_overlap)

                    for i in range(len(predictors_cuts)):
                        predictors_path.append(sound[:-6])
                        predictors.append(predictors_cuts[i])
                        target.append(target_cuts[i])

                else:
                    # if args.audio_visual:
                    #     image_predictors.append(image)
                    predictors_path.append(sound[:-6])
                    predictors.append(stft)
                    target.append(label)

                count += 1
                if args.num_data is not None and count >= args.num_data:
                    break

        return predictors, predictors_path, target

    train_folder = os.path.join(args.input_path, 'L3DAS23_Task2_train')
    test_folder = os.path.join(args.input_path, 'L3DAS23_Task2_dev')
    #testeval_folder = os.path.join(args.input_path, 'L3DAS22_Task2_test_w_labels')

    predictors_train, predictors_path_train, target_train = process_folder(train_folder, args)
    predictors_test, predictors_path_test,  target_test = process_folder(test_folder, args)
    #predictors_testeval, predictors_path_eval, target_testeval = process_folder(testeval_folder, args)

    predictors_test = [np.array(predictors_test), predictors_path_test]
    target_test = np.array(target_test)

    #split train set into train and development
    split_point = int(len(predictors_train) * args.train_val_split)
    predictors_training = [predictors_train[:split_point], predictors_path_train[:split_point]]    #attention: changed training names
    target_training = target_train[:split_point]
    predictors_validation = [predictors_train[split_point:], predictors_path_train[split_point:]]
    target_validation = target_train[split_point:]

    #save numpy matrices into pickle files
    print ('Saving files')
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path,'task2_predictors_train.pkl'), 'wb') as f:
        pickle.dump(predictors_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_predictors_validation.pkl'), 'wb') as f:
        pickle.dump(predictors_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_predictors_test.pkl'), 'wb') as f:
        pickle.dump(predictors_test, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_train.pkl'), 'wb') as f:
        pickle.dump(target_training, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_validation.pkl'), 'wb') as f:
        pickle.dump(target_validation, f, protocol=4)
    with open(os.path.join(args.output_path,'task2_target_test.pkl'), 'wb') as f:
        pickle.dump(target_test, f, protocol=4)

    print ('Matrices successfully saved')
    print ('Training set shape: ', np.array(predictors_training[0]).shape, np.array(target_training).shape)
    print ('Validation set shape: ', np.array(predictors_validation[0]).shape, np.array(target_validation).shape)
    print ('Test set shape: ', np.array(predictors_test[0]).shape, np.array(target_test).shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o
    mnt_path = '/mnt/media/christian/Datasets/'
    parser.add_argument('--task', type=int,
                        help='task to be pre-processed')
    parser.add_argument('--audio_visual', type=bool, default=True,
                        help='whether to consider images or not (audio-visual track or audio-only track)')
    parser.add_argument('--input_path', type=str, default=mnt_path+'DATASETS/Task1',
                        help='directory where the dataset has been downloaded')
    parser.add_argument('--output_path', type=str, default=mnt_path+'DATASETS/processed',
                        help='where to save the numpy matrices')
    #processing type
    parser.add_argument('--train_val_split', type=float, default=0.7,
                        help='perc split between train and validation sets')
    parser.add_argument('--num_mics', type=int, default=1,
                        help='how many ambisonics mics (1 or 2)')
    parser.add_argument('--num_data', type=int, default=None,
                        help='how many datapoints per set. 0 means all available data')
    #task1 only parameters
    # the following parameters produce 2-seconds waveform frames without overlap,
    # use only the train100 training set.
    parser.add_argument('--training_set', type=str, default='train100',
                        help='which training set: train100, train360 or both')
    parser.add_argument('--segmentation_len', type=float, default=None,
                        help='length of segmented frames in seconds')
    #task2 only parameters
    #the following stft parameters produce 8 stft fframes per each label frame
    #if label frames are 100msecs, stft frames are 12.5 msecs
    #data-points are segmented into 15-seconde windows (150 target frames, 150*8 stft frames)
    parser.add_argument('--frame_len', type=int, default=100,
                        help='frame length for SELD evaluation (in msecs)')
    parser.add_argument('--stft_nperseg', type=int, default=512,
                        help='num of stft frames')
    parser.add_argument('--stft_noverlap', type=int, default=112,
                        help='num of overlapping samples for stft')
    parser.add_argument('--stft_window', type=str, default='hamming',
                        help='stft window_type')
    parser.add_argument('--output_phase', type=str, default='False',
                        help='concatenate phase channels to stft matrix')
    parser.add_argument('--predictors_len_segment', type=int, default=None,
                        help='number of segmented frames for stft data')
    parser.add_argument('--target_len_segment', type=int, default=None,
                        help='number of segmented frames for stft data')
    parser.add_argument('--segment_overlap', type=float, default=None,
                        help='overlap factor for segmentation')
    parser.add_argument('--pad_length', type=float, default=4.792,
                        help='length of signal padding in seconds')
    parser.add_argument('--ov_subsets', type=str, default='["ov1", "ov2", "ov3"]',
                        help='should be a list of strings. Can contain ov1, ov2 and/or ov3')
    parser.add_argument('--no_overlaps', type=str, default='False',
                        help='should be a list of strings. Can contain ov1, ov2 and/or ov3')


    args = parser.parse_args()

    args.output_phase = eval(args.output_phase)
    args.ov_subsets = eval(args.ov_subsets)
    args.no_overlaps = eval(args.no_overlaps)

    if args.task == 1:
        preprocessing_task1(args)
    elif args.task == 2:
        preprocessing_task2(args)
