import ast
from datetime import datetime, timedelta
import librosa
import soundfile as sf
import numpy as np
import os
import random
from sklearn.preprocessing import LabelBinarizer
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from data_loader import TestSet
from model import Discriminator, Generator, Verify_CNN, Encoder, GeneratorTop
from preprocess import ALPHA, FRAMES, FFTSIZE, SHIFTMS
from utility import Normalizer, speakers, pad_mcep, synthesis_from_mcep
class Solver(object):
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader
        self.num_spk = config.num_spk

        if config.dataset == 'VCC2016':
            self.sample_rate = 16000
        else:
            self.sample_rate = 22050

        self.stop = False
        self.switch = None
        self.lambda_cyc = config.lambda_cyc
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        #self.lambda_v = (2/(1+np.exp(-10*(config.num_iters-1)))-1)
        self.lambda_v = 0.3

        # Training configurations.
        self.data_dir = config.data_dir
        self.test_dir = config.test_dir
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters
        self.trg_speaker = ast.literal_eval(config.trg_speaker)
        self.src_speaker = config.src_speaker

        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.spk_enc = LabelBinarizer().fit(speakers)

        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.start_testmodel_num = config.start_testmodel_num
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.v_epoch = 0 # 子网络训练轮次
        #self.veritfymodel_path = os.path.join(self.model_save_dir, '8000-V.ckpt')

        if config.mode == 'train':
            self.build_model()
        elif config.mode == 'convert' or config.mode == 'eval':
            self.load_model()

        # Only use tensorboard in train mode.
        if self.use_tensorboard and config.mode == 'train':
            self.build_tensorboard()

    def load_model(self):
        self.G = Generator(num_speakers=self.num_spk)
        self.test_iters = int(input("请输入加载的生成器模型轮次："))
        g_model_path = os.path.join(self.model_save_dir, f"{self.test_iters}-G.ckpt")

        if not os.path.exists(g_model_path):
            print("没有找到该预训练模型，请检查路径或选择从头开始训练。")
            return False

        self.G.load_state_dict(torch.load(g_model_path, map_location=lambda storage, loc: storage))
        self.print_network(self.G, 'G')  # 打印网络结构
        self.G.to(self.device)  # 设置gpu

    def load_verifier_model(self):
        self.v = Verify_CNN(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, num_speaker=4, verityswitch2=3)
        verify_switch = int(input("1.重新预训练子网络; 2.加载本地子网络模型; 3.不进行子网络预训练，直接开始对抗训练"))
        if verify_switch == 1:
            self.epoch_v = 0
            self.train_veritfy = True

        elif verify_switch == 3:
            self.epoch_v = 0
            self.v_epoch = 0
            self.v.to(self.device)
            self.train_veritfy = False

        elif verify_switch == 2:
            self.train_veritfy = False
            self.epoch_v = int(input("请输入加载的子网络模型轮次："))
            self.veritfymodel_path = os.path.join(self.model_save_dir, f"{self.epoch_v}-v.ckpt")
            path = self.veritfymodel_path
            if os.path.exists(path):
                self.v.load_state_dict(torch.load(path))
                verify_switch2 = input("是否继续训练本地子网络模型?y/n").lower()
                if verify_switch2 == 'y':
                    self.train_veritfy = True
                else:
                    self.v.to(self.device)
            else:
                print("没有子网络模型，请检查路径或选择从头开始训练。")
                return False
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), self.g_lr, [self.beta1, self.beta2])  # 训练经过两次下采样，一次降维后语音特征的模型
        return True

    def load_pretrained_models(self):
        train_from_scratch = input("是否从头开始训练生成器和判别器？y/n：").lower()
        if train_from_scratch == "n":
            epoch_num = int(input("请输入加载的模型轮次："))
            self.start_testmodel_num = epoch_num
            g_model_path = os.path.join(self.model_save_dir, f"{epoch_num}-G.ckpt")
            d_model_path = os.path.join(self.model_save_dir, f"{epoch_num}-D.ckpt")

            if not os.path.exists(g_model_path) or not os.path.exists(d_model_path):
                print("没有找到该预训练模型，请检查路径或选择从头开始训练。")
                return False

            self.G.load_state_dict(torch.load(g_model_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(d_model_path, map_location=lambda storage, loc: storage))
        return True

    def build_model(self):
        self.encoder = Encoder(num_speakers=self.num_spk).cuda()
        self.generator_top = GeneratorTop(num_speakers=self.num_spk).cuda()
        self.G = Generator(num_speakers=self.num_spk)
        self.D = Discriminator(num_speakers=self.num_spk)
        switch = input("是否进行子网络对抗训练？y/n")
        if switch.lower() == "y":
            self.switch = True
            if not self.load_verifier_model():
                self.stop = True
                return

        elif switch.lower() == "n":
            self.train_veritfy = False
            self.switch = False

        if not self.load_pretrained_models():
            self.stop = True
            return

        #self.g_optimizer = torch.optim.Adam(params=[{'params':self.encoder.parameters()},{'params':self.generator_top.parameters()}], lr = self.g_lr, betas = [self.beta1, self.beta2])#对应生成器的优化器
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=[self.beta1, self.beta2])  # 对应生成器的优化器
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(f"* ({name}) Number of parameters: {num_params}.")

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """
            Decay learning rates of the generator and discriminator.
        """

        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def train_veritfymodel(self):
        self.v.to(self.device)
        self.v.verifyingswitch2 = 3
        data_iter = iter(self.data_loader)
        num_loss = 0
        num_acc = 0
        self.v_epoch = 8000
        for j in range(self.v_epoch):#分类器训练
            try:
                x_real, speaker_idx_org, label_org = next(data_iter)

            except:
                data_iter = iter(self.data_loader)
                x_real, speaker_idx_org, label_org = next(data_iter)
            x_real = x_real.to(self.device)
            #label_org = label_org.to(self.device)
            speaker_idx_org = speaker_idx_org.to(self.device)
            embadding, size = self.encoder(x_real)#提取进行两次下采样和一次降维后的数据，结合目标标签训练这个数据,即编码结构的输出
            lossverity, acc = self.compute_loss_accuracy(embadding.detach(), speaker_idx_org)
            self.v_optimizer.zero_grad()
            lossverity.backward()
            self.v_optimizer.step()
            num_loss += lossverity
            num_acc += acc
            if (j + 1) % 100 == 0:
                num_loss = num_loss.item()
                print("Epoch: {}/{}, Loss: {:.4f}, Acc: {:.4f}".format(j + 1, self.v_epoch, num_loss / j, num_acc / j))
        self.veritfymodel_path = os.path.join(self.model_save_dir, f"{self.epoch_v + self.v_epoch}-v.ckpt")
        torch.save(self.v.state_dict(), self.veritfymodel_path)
        print("veritfy model finished training")

    def train(self):
        if self.stop == True:
            return
        if self.train_veritfy == True:
            self.train_veritfymodel()
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_iters = 0
        # if self.resume_iters:
        #     print(f'Resume at step {self.resume_iters}...')
        #     start_iters = self.resume_iters
        #     self.restore_model(self.resume_iters)

        norm = Normalizer()
        data_iter = iter(self.data_loader)

        g_adv_optim = 0
        g_adv_converge_low = True  # Check which direction `g_adv` is converging (init as low).
        g_rec_optim = 0
        g_rec_converge_low = True  # Check which direction `g_rec` is converging (init as low).
        g_tot_optim = 0
        g_tot_converge_low = True  # Check which direction `g_tot` is converging (init as low).

        print('\n* Start training...\n')
        start_time = datetime.now()

        for i in range(start_iters, self.num_iters):
            try:
                x_real, speaker_idx_org, label_org = next(data_iter)
                #print (x_real.shape,speaker_idx_org.shape,label_org.shape)

            except:
                data_iter = iter(self.data_loader)
                x_real, speaker_idx_org, label_org = next(data_iter)

            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            speaker_idx_trg = speaker_idx_org[rand_idx]

            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)  # Original domain one-hot labels.
            label_trg = label_trg.to(self.device)  # Target domain one-hot labels.
            speaker_idx_org = speaker_idx_org.to(self.device)  # Original domain labels.
            speaker_idx_trg = speaker_idx_trg.to(self.device)  # Target domain labels.

            """
                Discriminator training.
            """
            CELoss = nn.CrossEntropyLoss()
            g_loss_item = {}
            d_loss_item = {}
            v_loss_item = {}
            v_acc_item = {}

            # Loss: st-adv.
            out_r = self.D(x_real, label_org, label_trg)
            # print(out_r.shape)
            x_fake = self.G(x_real, label_org, label_trg)

            out_f = self.D(x_fake.detach(), label_org, label_trg)
            d_loss_adv = F.binary_cross_entropy_with_logits(input=out_f,
                                                            target=torch.ones_like(out_f, dtype=torch.float)) + \
                         F.binary_cross_entropy_with_logits(input=out_r,
                                                            target=torch.ones_like(out_r, dtype=torch.float))#原来fake那边是oneslike
            # Loss: gp.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # print (alpha.shape)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            # print (x_hat.shape)
            out_src = self.D(x_hat, label_org, label_trg)
            # print (out_src.shape)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Totol loss: st-adv + lambda_gp * gp.
            d_loss = d_loss_adv + self.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            d_loss_item['D/d_loss_adv'] = d_loss_adv.item()  # 判别器鉴定给出的语音是否为生成器创造
            d_loss_item['D/d_loss_gp'] = d_loss_gp.item()  # 梯度下降损失，待研究grad penalty
            d_loss_item['D/d_loss'] = d_loss.item()  # 上面两个损失的相加


            if self.switch == True:
                # 训练子网络 计算更新编码结构前的损失 编码结构更新前的子网络识别正确率
                embadding, size = self.encoder(x_real)  # 提取进行两次下采样和一次降维后的数据，结合目标标签训练这个数据,即编码结构的输出
                # speakerloss, SpeakerCorrect = self.compute_loss_accuracy(embadding.detach(), speaker_idx_org)
                #
                # print("SpeakerLoss: {:.5f}, SpeakerACC: {:.3f}".format(speakerloss.item(), SpeakerCorrect))
                #
                # self.v_optimizer.zero_grad()
                # speakerloss.backward()  # 只优化子网络
                # self.v_optimizer.step()
                #
                # v_loss_item['V/be_spk_loss'] = speakerloss.item()  # 说话人识别损失
                # v_acc_item['V/be_spk_acc'] = SpeakerCorrect  # 子网络识别正确率

            """
                Generator training.
            """
            if (i + 1) % self.n_critic == 0:
                # 原训练方法
                if self.switch == False:
                    # Loss: st-adv (original-to-target).
                    x_fake = self.G(x_real, label_org, label_trg)

                    g_out_src = self.D(x_fake, label_org, label_trg)
                    g_loss_adv = F.binary_cross_entropy_with_logits(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))

                    # Loss: cyc (target-to-original).
                    x_rec = self.G(x_fake, label_trg, label_org)  # 伪造语音转成原语音
                    g_loss_rec = F.l1_loss(x_rec, x_real)

                    # Loss: id (original-to-original).
                    x_fake_id = self.G(x_real, label_org, label_org)
                    g_loss_id = F.l1_loss(x_fake_id, x_real)

                    # Total loss: st-adv + lambda_cyc * cyc + lambda_id * id.
                    # Only include Identity mapping before 10k iterations.
                    if (i + self.start_testmodel_num + 1) < 10 ** 4 + 2:
                        g_loss = g_loss_adv \
                                 + self.lambda_cyc * g_loss_rec \
                                 + self.lambda_id * g_loss_id \

                    else:
                        g_loss = g_loss_adv + self.lambda_cyc * g_loss_rec

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                # 添加子网络训练方法
                elif self.switch == True:
                    # Loss: st-adv (original-to-target).
                    # 生成目标语音 a-b
                    x_fake = self.G(x_real, label_org, label_trg)#生成目标语音
                    #x_fake = self.generator_top(embadding, label_org, label_trg, size)  # 生成目标语音

                    # 计算 a-b 子网络识别损失和正确率
                    ae_speakerloss, ae_SpeakerCorrect = self.compute_loss_accuracy(embadding, speaker_idx_org)
                    v_loss_item['V/ae_spk_loss'] = ae_speakerloss.item()  # 说话人识别损失
                    v_acc_item['V/ae_spk_acc'] = ae_SpeakerCorrect  # 子网络识别正确率

                    g_out_src = self.D(x_fake, label_org, label_trg)
                    g_loss_adv = F.binary_cross_entropy_with_logits(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))

                    # Loss: cyc (target-to-original).
                    # 伪造语音生成原语音 b-a
                    fake_embadding, size = self.encoder(x_fake)
                    x_rec = self.G(x_fake, label_trg, label_org)  # 伪造语音转成原语音
                    #x_rec = self.generator_top(fake_embadding, label_trg, label_org, size)  # 生成目标语音

                    # 计算 b-a 子网络识别损失和正确率
                    fake_ae_speakerloss, fake_ae_SpeakerCorrect = self.compute_loss_accuracy(fake_embadding, speaker_idx_trg)
                    v_loss_item['V/fake_ae_spk_loss'] = fake_ae_speakerloss.item()  # 说话人识别损失
                    v_acc_item['V/fake_ae_spk_acc'] = fake_ae_SpeakerCorrect  # 子网络识别正确率

                    g_loss_rec = F.l1_loss(x_rec, x_real)
                    # if (i + 1) < 10 ** 4 + 2:
                    #     g_loss_rec = F.l1_loss(x_rec, x_real) - u * fake_ae_speakerloss
                    # else:
                    #     g_loss_rec = F.l1_loss(x_rec, x_real)

                    # Loss: id (original-to-original).
                    # 原语音生成原语音 a-a
                    fake_id_embadding, size = self.encoder(x_real)
                    x_fake_id = self.G(x_real, label_org, label_org)
                    #x_fake_id = self.generator_top(fake_id_embadding, label_org, label_org, size)  # 生成目标语音

                    # 计算 a-a 子网络识别损失和正确率
                    id_ae_speakerloss, id_ae_SpeakerCorrect = self.compute_loss_accuracy(fake_id_embadding, speaker_idx_org)
                    v_loss_item['V/id_ae_spk_loss'] = id_ae_speakerloss.item()  # 说话人识别损失
                    v_acc_item['V/id_ae_spk_acc'] = id_ae_SpeakerCorrect  # 子网络识别正确率

                    g_loss_id = F.l1_loss(x_fake_id, x_real)
                    # if (i + 1) < 10 ** 4 + 2:
                    #     g_loss_id = F.l1_loss(x_fake_id, x_real) - (u / 3) * id_ae_speakerloss
                    # else:
                    #     g_loss_id = F.l1_loss(x_fake_id, x_real)

                    v_g_loss = self.lambda_v * ae_speakerloss + (self.lambda_v / 2) * fake_ae_speakerloss + (self.lambda_v / 3) * id_ae_speakerloss
                    # Total loss: st-adv + lambda_cyc * cyc + lambda_id * id + lambda_v * speakerloss.
                    # Only include Identity mapping before 10k iterations.
                    if (i + self.start_testmodel_num + 1) < 10 ** 4 + 2:
                        print('执行10000步之前')
                        g_loss = g_loss_adv \
                                 + self.lambda_cyc * g_loss_rec \
                                 + self.lambda_id * g_loss_id \
                                 #- v_g_loss
                    else:
                        print('执行10000步以后')
                        g_loss = g_loss_adv + self.lambda_cyc * g_loss_rec #- v_g_loss

                    # 生成器优化梯度
                    self.v.lock_grad()
                    self.reset_grad()
                    # (g_loss - v_g_loss).backward(retain_graph=True)
                    # g_loss.backward()
                    (g_loss - v_g_loss).backward()
                    self.g_optimizer.step()
                    self.v.acquire_grad()

                    # 计算v_loss，优化子网络
                    ae_speakerloss, ae_SpeakerCorrect = self.compute_loss_accuracy(embadding.detach(), speaker_idx_org)
                    fake_ae_speakerloss, fake_ae_SpeakerCorrect = self.compute_loss_accuracy(fake_embadding.detach(), speaker_idx_trg)
                    id_ae_speakerloss, id_ae_SpeakerCorrect = self.compute_loss_accuracy(fake_id_embadding.detach(), speaker_idx_org)
                    #v_loss = self.lambda_v * ae_speakerloss + (self.lambda_v / 2) * fake_ae_speakerloss + (self.lambda_v / 3) * id_ae_speakerloss
                    v_loss = ae_speakerloss + fake_ae_speakerloss + id_ae_speakerloss
                    speakercorrect = ae_SpeakerCorrect + fake_ae_SpeakerCorrect + id_ae_SpeakerCorrect

                    v_loss_item['V/v_g_loss'] = v_g_loss.item() # 加入到生成器对抗训练的子网络损失
                    v_loss_item['V/v_loss'] = v_loss.item()  # 说话人识别总损失
                    v_acc_item['V/v_acc'] = speakercorrect / 3 # 子网络识别总体平均正确率

                    # 子网络优化梯度
                    self.v_optimizer.zero_grad()
                    v_loss.backward()
                    self.v_optimizer.step()

                    # self.G.encoder = self.encoder
                    # self.G.generator_top = self.generator_top

                # Check convergence direction of losses.
                if (i + 1) == 20 * (10 ** 3):  # Update optims at 20k iterations.
                    g_adv_optim = g_loss_adv
                    g_rec_optim = g_loss_rec
                    g_tot_optim = g_loss
                if (i + 1) == 70 * (10 ** 3):  # Check which direction optims have gone over 70k iters.
                    if g_loss_adv > g_adv_optim:
                        g_adv_converge_low = False
                    if g_loss_rec > g_rec_optim:
                        g_rec_converge_low = False
                    if g_loss > g_tot_optim:
                        g_tot_converge_low = False

                    print('* CONVERGE DIRECTION')
                    print(f'adv_loss low: {g_adv_converge_low}')
                    print(f'g_rec_loss los: {g_rec_converge_low}')
                    print(f'g_loss loq: {g_tot_converge_low}')

                # Update loss for checkpoint saving.
                if (i + 1) > 75 * (10 ** 3):
                    if g_tot_converge_low:
                        if (g_loss_adv < g_adv_optim and abs(
                                g_loss_adv - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')
                    elif not g_tot_converge_low:
                        if (g_loss_adv > g_adv_optim and abs(
                                g_loss_adv - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')

                    if g_adv_converge_low:
                        if g_loss_adv < g_adv_optim:
                            g_adv_optim = g_loss_adv
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')
                    elif not g_adv_converge_low:
                        if g_loss_adv < g_adv_optim:
                            g_adv_optim = g_loss_adv
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')

                    if g_rec_converge_low:
                        if g_loss_rec < g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')
                    elif not g_rec_converge_low:
                        if g_loss_rec > g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')

                    if g_tot_converge_low:
                        if g_loss < g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')
                    elif not g_tot_converge_low:
                        if g_loss > g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')

                g_loss_item['G/g_loss_adv'] = g_loss_adv.item() # 生成器对抗损失，越小越好
                g_loss_item['G/g_loss_rec'] = g_loss_rec.item() # 生成目标再转回来
                g_loss_item['G/g_loss_id'] = g_loss_id.item() # 自己生成自己
                g_loss_item['G/g_loss'] = g_loss.item()

            # if (self.switch == True) and ((i + 1) % self.n_critic != 0):
            #     # 训练子网络 计算更新编码结构前的损失 编码结构更新前的子网络识别正确率
            #     speakerloss, SpeakerCorrect = self.compute_loss_accuracy(embadding.detach(), speaker_idx_org)
            #
            #     print("SpeakerLoss: {:.5f}, SpeakerACC: {:.3f}".format(speakerloss.item(), SpeakerCorrect))
            #
            #     self.v_optimizer.zero_grad()
            #     speakerloss.backward()  # 只优化子网络
            #     self.v_optimizer.step()


            # Print training information.
            if (i + 1) % self.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[: -7]
                self.log_print(et, i, g_loss_item)
                self.log_print(et, i, d_loss_item)
                if self.switch == True:
                    self.log_print(et, i, v_loss_item)
                    self.log_print(et, i, v_acc_item)

                if self.use_tensorboard:
                    for tag, value in g_loss_item.items():
                        self.logger.scalar_summary(tag, value, self.start_testmodel_num + i + 1)
                    for tag, value in d_loss_item.items():
                        self.logger.scalar_summary(tag, value, self.start_testmodel_num + i + 1)
                    if self.switch == True:
                        for tag, value in v_loss_item.items():
                            self.logger.scalar_summary(tag, value, self.start_testmodel_num + i + 1)
                        for tag, value in v_acc_item.items():
                            self.logger.scalar_summary(tag, value, self.start_testmodel_num + i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    d, speaker = TestSet(self.test_dir, self.sample_rate).test_data()
                    original = random.choice([x for x in speakers if x != speaker])
                    target = random.choice([x for x in speakers if x != speaker])
                    label_o = self.spk_enc.transform([original])[0]
                    label_t = self.spk_enc.transform([target])[0]
                    label_o = np.asarray([label_o])
                    label_t = np.asarray([label_t])

                    for filename, content in d.items():
                        f0 = content['f0']
                        ap = content['ap']
                        mcep_norm_pad = pad_mcep(content['mcep_norm'], FRAMES)

                        convert_result = []
                        for start_idx in range(0, mcep_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                            one_seg = mcep_norm_pad[:, start_idx: start_idx + FRAMES]

                            one_seg = torch.FloatTensor(one_seg).to(self.device)
                            one_seg = one_seg.view(1, 1, one_seg.size(0), one_seg.size(1))
                            o = torch.FloatTensor(label_o)
                            t = torch.FloatTensor(label_t)
                            one_seg = one_seg.to(self.device)
                            o = o.to(self.device)
                            t = t.to(self.device)
                            one_set_return = self.G(one_seg, o, t).data.cpu().numpy()
                            # loss = self.veritfy_loss(x1, speaker_idx_org)
                            # print("验证是否去除语音特征的损失值：", loss)
                            # one_set_return = self.G(one_seg, o, t).data.cpu().numpy()
                            one_set_return = np.squeeze(one_set_return)
                            one_set_return = norm.backward_process(one_set_return, target)
                            convert_result.append(one_set_return)

                        convert_con = np.concatenate(convert_result, axis=1)
                        convert_con = convert_con[:, 0: content['mcep_norm'].shape[1]]
                        contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)
                        f0_converted = norm.pitch_conversion(f0, speaker, target)
                        wav = synthesis_from_mcep(f0_converted, contigu, ap, self.sample_rate, FFTSIZE, SHIFTMS, ALPHA)

                        name = f'{speaker}-{target}_iter{i + self.start_testmodel_num + 1}_{filename}'
                        path = os.path.join(self.sample_dir, name)
                        print(f'[SAVE]: {path}')
                        sf.write(path, wav, self.sample_rate)
                        # librosa.output.write_wav(path, wav, self.sample_rate)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(self.start_testmodel_num + i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(self.start_testmodel_num + i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                if self.switch == True:
                    V_path = os.path.join(self.model_save_dir, '{}-v.ckpt'.format(self.epoch_v + self.v_epoch + i + 1))
                    torch.save(self.v.state_dict(), V_path)
                print(f'Save model checkpoints into {self.model_save_dir}...')

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print(f'Decayed learning rates, g_lr: {g_lr}, d_lr: {d_lr}.')

    def gradient_penalty(self, y, x):
        """
            Compute gradient penalty: (L2_norm(dy / dx) - 1) ** 2.
            (Differs from the paper.)
        """

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def veritfy_loss(self, out, x):
        #loss_fun = nn.CrossEntropyLoss()
        loss_fun = nn.L1Loss()
        x = F.one_hot(x, num_classes=4)
        #print("x:",x)
        loss = loss_fun(out, x)
        return loss

    # KL散度损失函数
    def kl_divergence_loss(self, output, real):
        #real = F.one_hot(real, num_classes=4).float()
        output_probs = F.log_softmax(output, dim=1)
        #print('output_prob:',output_probs)
        real_probs = torch.softmax(real, dim=1)
        #print('real_prob:',real_probs)
        loss = F.kl_div(output_probs, real_probs, reduction='batchmean')
        return loss

    # 返回子网络识别损失值和正确率
    def compute_loss_accuracy(self, embadding, speaker_id_org):
        # 生成子网络识别结果
        sp_pred = self.v(embadding)

        #print('sp_pred:',sp_pred)
        # 计算子网络识别正确率
        ae_correct = torch.argmax(sp_pred, 1).cuda()
        speaker_correct = ((ae_correct == speaker_id_org).sum().float()) / len(speaker_id_org)

        #print("speaker_id_org:",speaker_id_org)
        #print("pred:",ae_correct)

        # 计算 kl 散度损失
        speaker_id_org = F.one_hot(speaker_id_org, num_classes=4).float()
        speaker_loss = self.kl_divergence_loss(sp_pred, speaker_id_org)

        return speaker_loss, speaker_correct

    # 输出损失值信息
    def log_print(self, et, i, info):
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, self.start_testmodel_num + i + 1, self.start_testmodel_num + self.num_iters)
        for tag, value in info.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def save_optim_checkpoints(self, g_name, d_name, type_saving):
        G_path = os.path.join(self.model_save_dir, g_name)
        D_path = os.path.join(self.model_save_dir, d_name)
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print(f'Save {type_saving} optimal model checkpoints into {self.model_save_dir}...')

    def restore_model(self, resume_iters):
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        #D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        #self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def convert(self):
        """
            Convertion.
        """

        #self.restore_model(self.test_iters)
        norm = Normalizer()

        d, speaker = TestSet(self.test_dir, self.sample_rate).test_data(self.src_speaker)  # 相同的特征提取和读取方式
        #print("d-------------",d)
        #print("speaker-------",speaker)
        targets = self.trg_speaker
        sum = 0
        count = 0
        sum_target = 0
        count_target = 0
        for target in targets:
            if target != self.src_speaker:
                ta, sp = TestSet(self.test_dir, self.sample_rate).test_data(target)
                ta_content = get_mcep(ta)
            print(f'* Target: {target}')
            assert target in speakers
            label_o = self.spk_enc.transform([self.src_speaker])[0]
            label_t = self.spk_enc.transform([target])[0]
            label_o = np.asarray([label_o])
            label_t = np.asarray([label_t])

            with torch.no_grad():
                for filename, content in d.items():
                    f0 = content['f0']
                    ap = content['ap']
                    mcep_norm_pad = pad_mcep(content['mcep_norm'], FRAMES)
                    #print(content['mcep_norm'].shape)
                    #print(mcep_norm_pad.shape)
                    #print(mcep_norm_pad)

                    convert_result = []
                    for start_idx in range(0, mcep_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                        one_seg = mcep_norm_pad[:, start_idx: start_idx + FRAMES]
                        #print(one_seg.shape)
                        one_seg = torch.FloatTensor(one_seg).to(self.device)
                        #print(one_seg.shape)
                        one_seg = one_seg.view(1, 1, one_seg.size(0), one_seg.size(1))
                        #print(one_seg.shape)
                        o = torch.FloatTensor(label_o)
                        t = torch.FloatTensor(label_t)
                        one_seg = one_seg.to(self.device)
                        o = o.to(self.device)
                        t = t.to(self.device)
                        one_set_return = self.G(one_seg, o, t).data.cpu().numpy()
                        one_set_return = np.squeeze(one_set_return)
                        one_set_return = norm.backward_process(one_set_return, target)
                        convert_result.append(one_set_return)

                    convert_con = np.concatenate(convert_result, axis=1)
                    # print(convert_con.shape)
                    if target == self.src_speaker:
                        convert_con = convert_con[:, 0: content['or_mcep'].shape[1]]
                        # convert_con1 = norm.forward_process(convert_con,target)
                        # print(content['or_mcep'])
                        # print(convert_con1)
                        a = mcd(content['or_mcep'], convert_con)
                    elif target != self.src_speaker:
                        convert_con1 = convert_con#伪造的目标语音
                        convert_con = convert_con[:, 0: content['or_mcep'].shape[1]]#维度转换后的原目标语音，不用归一化了
                        print('ta_content:', ta_content[count].shape)
                        if len(convert_con1[1]) > len(ta_content[count][1]):
                            convert_con1 = convert_con1[:, 0: ta_content[count].shape[1]]
                        elif len(convert_con1[1]) < len(ta_content[count][1]):
                            ta_content[count] = ta_content[count][:, 0: convert_con1.shape[1]]
                        # print(convert_con1.shape)
                        # print(ta_content[count].shape)
                        a = mcd(ta_content[count], convert_con1)
                    # a = np.mean((10 * np.sqrt(2)) / np.log(10) * a)
                    a = np.mean(10 * np.log10(a))
                    print(f'*转换后的{target}和原来的{target}特征距离(dB)：', a)
                    sum += a
                    count += 1
                    contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)
                    f0_converted = norm.pitch_conversion(f0, speaker, target)
                    wav = synthesis_from_mcep(f0_converted, contigu, ap, self.sample_rate, FFTSIZE, SHIFTMS, ALPHA)

                    name = f'{speaker}-{target}_iter{self.test_iters}_{filename}'
                    path = os.path.join(self.result_dir, name)
                    print(f'[SAVE]: {path}')
                    #librosa.output.write_wav(path, wav, self.sample_rate)
                    sf.write(path, wav, self.sample_rate)
            ad = sum / count
            print(ad)
            sum = 0
            count = 0
            count_target += 1
            sum_target += ad
        print('所有目标转化音色的平均距离(dB）',sum_target/count_target)
    '''
    def eval(self):
        self.restore_model(self.test_iters)
        self.build_tensorboard()
        self.train()'''

    def eval(self):
        self.restore_model(self.test_iters)
        self.build_tensorboard()
        self.G.verifyingswitch = True
        self.v = Verify_CNN(dim_in=256, dim_out=512, kernel_size=5, stride=1, padding=2, num_speaker=4,
                            verityswitch2=1)
        path = self.veritfymodel_path
        self.v.load_state_dict(torch.load(path))
        self.v.to(self.device)
        #self.v.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))
        # self.v.lock_grad()
        # self.v.acquire_grad()
        self.v.verifyingswitch2 = 3
        self.switch = True
        self.G.verifyCNN = self.v
        #self.G.verifyCNN.lock_grad()
        self.G.normalswitch = False
        self.train()


def mcd(target_mcep, converted_mcep):
    return np.sqrt(np.sum((target_mcep - converted_mcep) ** 2, axis=1))

def get_mcep(d):
    getmcep = []
    for filename, content in d.items():
        mcep_norm = content['or_mcep']
        getmcep.append(mcep_norm)
        #print(mcep_norm.shape)
        #print(mcep_norm)
    return getmcep




