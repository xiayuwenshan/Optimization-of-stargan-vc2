from data_loader import TestSet
from utility import Normalizer, speakers, cal_mcep,pad_mcep
from preprocess import ALPHA, FEATURE_DIM, FFTSIZE, FRAMES, SHIFTMS
import librosa
import os
from random import choice
import numpy as np


class TestSet_copy(object):
    def __init__(self, data_dir: str, sr: int):
        super(TestSet_copy, self).__init__()
        self.data_dir = data_dir
        self.norm = Normalizer()
        self.sample_rate = sr

    def choose(self):
        r = choice(speakers)
        return r

    def test_data(self, src_speaker=None):
        if src_speaker:
            r_s = src_speaker
        else:
            r_s = self.choose()

        p = os.path.join(self.data_dir, r_s)  # 拼接路径，测试目录+具体人的文件夹
        wavfiles = librosa.util.find_files(p, ext='wav')
        # print(wavfiles)
        # wavefiles是一个文件路径的列表，格式如下
        # ['路径/1号.wav', '路径/2号.wav'...]
        res = {}
        for f in wavfiles:
            filename = os.path.basename(f)  # 读取最后一层文件本身的名字
            wav, _ = librosa.load(f, sr=self.sample_rate, dtype=np.float64)
            f0, ap, mcep = cal_mcep(wav, self.sample_rate, FEATURE_DIM, FFTSIZE, SHIFTMS, ALPHA)
            # print(mcep)
            # print(mcep.shape)
            mcep_norm = self.norm.forward_process(mcep, r_s)

            if not res.__contains__(filename):
                res[filename] = {}
            res[filename]['mcep_norm'] = np.asarray(mcep_norm)
            res[filename]['f0'] = np.asarray(f0)
            res[filename]['ap'] = np.asarray(ap)
        return res, r_s

def mcd(target_mcep, converted_mcep):
    return np.mean(np.sqrt(np.sum((target_mcep - converted_mcep) ** 2, axis=1)))

def get_mcep(d):
    getmcep = []
    for filename, content in d.items():
        mcep_norm = content['mcep_norm']
        getmcep.append(mcep_norm)
        #print(mcep_norm.shape)
        #print(mcep_norm)
    return getmcep

def get_mcd():
    src = input("SF1或TM1")
    #src = input("TM1-SF1或TM1-TM1")
    d, sp = TestSet_copy('outputs/results/改进模型的测试结果', 16000).test_data(src)#改进模型生成的语音
    d1, sp1 = TestSet_copy('data\spk_test', 16000).test_data(src)#原语音
    d2, sp2 = TestSet_copy('outputs/results/原始模型的测试结果', 16000).test_data(src)#原始模型
    mcep = get_mcep(d)
    mcep1 = get_mcep(d1)
    mcep2 = get_mcep(d2)
    count = 0
    sum = 0
    sum2 = 0
    for i in range(len(mcep1)):
        mcep[i] = pad_mcep(mcep[i],FRAMES)
        mcep1[i] = pad_mcep(mcep1[i], FRAMES)
        mcep2[i] = pad_mcep(mcep2[i], FRAMES)
        print("mcep:",mcep[i].shape)
        print("mcep1",mcep1[i].shape)
        print("mcep2",mcep2[i].shape)
        count += 1
        sum += mcd(mcep[i],mcep1[i])
        sum2 += mcd(mcep2[i],mcep1[i])
    ad = sum / count
    ad2 = sum2 / count
    print(ad,ad2)

if __name__ == '__main__':
    get_mcd()