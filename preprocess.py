import argparse
from datetime import datetime
import glob
import librosa #需要pip install 安装 （https://pypi.org/project/librosa/0.7.0rc1/#files）
# https://github.com/deezer/spleeter/issues/419
import numpy as np
import os
import shutil

from utility import GenerateStatistics, speakers, cal_mcep

#指定好参数：--dataset VCC2016

FEATURE_DIM = 34
FRAMES = 128#维度
FFTSIZE = 1024
SPEAKERS_NUM = len(speakers)
CHUNK_SIZE = 1
EPSILON = 1e-10
SHIFTMS = 5.0
ALPHA = 0.42

def load_wavs(dataset: str, sr):
    """
        `data`: contains all audios file path. 
        `resdict`: contains all wav files.   
    """

    data = {}
    with os.scandir(dataset) as it:#装载数据
        for entry in it:
            if entry.is_dir():
                data[entry.name] = []
                with os.scandir(entry.path) as it_f:
                    for onefile in it_f:
                        if onefile.is_file():
                            data[entry.name].append(onefile.path)#定位到最后的目录之中
    print(f'* Loaded keys: {data.keys()}')
    resdict = {}

    cnt = 0
    for key, value in data.items():#data数据分为key：主人和value：信息所在路径，4乘162
        resdict[key] = {}

        for one_file in value: #预处理，突出高频信号，因为一般发音的话高频信号能表达更多有用的信息，预加重
            filename = one_file.split('/')[-1].split('.')[0] 
            newkey = f'{filename}'
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)#这一步加载了语音数据，sr：采样率16000 mono：单通道
            y, _ = librosa.effects.trim(wav, top_db=15)
            wav = np.append(y[0], y[1: ] - 0.97 * y[: -1])

            resdict[key][newkey] = wav
            print('.', end='')
            cnt += 1

    print(f'\n* Total audio files: {cnt}.')
    return resdict

def chunks(iterable, size):
    """
        Yield successive n-sized chunks from iterable.
    """

    for i in range(0, len(iterable), size):
        yield iterable[i: i + size]

def wav_to_mcep_file(dataset: str, sr: int, processed_filepath: str='./data/processed'):
    """
        Convert wavs to MCEPs feature using image representation.
    """

    shutil.rmtree(processed_filepath)
    os.makedirs(processed_filepath, exist_ok=True)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    print(f'* Total audio files: {allwavs_cnt}.')

    d = load_wavs(dataset, sr)#加载语音数据
    for one_speaker in d.keys():#每个人的全部数据，循环4次处理完
        values_of_one_speaker = list(d[one_speaker].values())#搞到列表里去
       
        for index, one_chunk in enumerate(chunks(values_of_one_speaker, CHUNK_SIZE)):
            wav_concated = [] 
            temp = one_chunk.copy()

            for one in temp:
                wav_concated.extend(one)
            wav_concated = np.array(wav_concated)

            f0, ap, mcep = cal_mcep(wav_concated, sr, FEATURE_DIM, FFTSIZE, SHIFTMS, ALPHA)#生成这三个特征，mcep约等于mfcc
            #基频特征f0，aperiodic参数(由f0与频谱包络sp算出)以及mcep特征
            # MFCC：连续语音--预加重--加窗分帧--FFT--MEL滤波器组--对数运算--DCT
            #这里的操作在utility.py中，都是通过包实现的
            newname = f'{one_speaker}_{index}'
            file_path_z = os.path.join(processed_filepath, newname)#把这些特征保存到这个路径里面
            np.savez(file_path_z, f0=f0, mcep=mcep)
            print(f'[SAVE]: {file_path_z}')

            for start_idx in range(0, mcep.shape[1] - FRAMES + 1, FRAMES):#将特征切片，每128个特征分成一个维度
                one_audio_seg = mcep[:, start_idx: start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:
                    temp_name = f'{newname}_{start_idx}'
                    filePath = os.path.join(processed_filepath, temp_name)
                    np.save(filePath, one_audio_seg)
                    print(f'[SAVE]: {filePath}.npy')
            
if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics.')
    
    input_dir = './data/spk'
    output_dir = './data/processed'

    dataset_default = 'VCC2016'

    parser.add_argument('--dataset', type=str, default=dataset_default, choices=['VCC2016', 'VCC2018'], 
        help='Available datasets: VCC2016 and VCC2018 (Default: VCC2016).')
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Directory of input data.')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Directory of processed data.')
    
    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir

    os.makedirs(output_dir, exist_ok=True)

    """
        Sample rate:
            VCC2016: 16000 Hz
            VCC2018: 22050 Hz
    """
    if argv.dataset == 'VCC2016':
        sample_rate = 16000
    else:
        sample_rate = 22050

    wav_to_mcep_file(input_dir, sample_rate, processed_filepath=output_dir)#预处理操作，波形到特征

    generator = GenerateStatistics(output_dir)
    generator.generate_stats()
    generator.normalize_dataset()
    end = datetime.now()
    
    print(f"* Duration: {end-start}.")
