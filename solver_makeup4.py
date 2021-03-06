import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models.vgg as models
from tensorboardX import SummaryWriter

import os
import time
import datetime

import tools.plot as plot_fig
import net, network2
from ops.histogram_matching import *
from ops.loss_added import GANLoss

class Solver_makeupGAN4(object):
    def __init__(self, data_loaders, config, dataset_config):
        # gpu
        self.multi_gpu = config.multi_gpu
        self.gpu_ids = config.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpu_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dataloader
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.e_lr = config.E_LR
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 200
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.whichG = config.whichG
        self.norm = config.norm

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step

        #training setting
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders[0]
        self.data_loader_test = data_loaders[1]

        # Model hyper-parameters
        self.img_size = config.img_size
        self.e_conv_dim = config.e_conv_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye

        # Hyper-parameteres
        self.lambda_idt = config.lambda_idt
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_skin_1 = config.lambda_his_skin_1
        self.lambda_his_skin_2 = config.lambda_his_skin_2
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_vgg = config.lambda_vgg

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.cls = config.cls_list
        self.content_layer = config.content_layer
        self.direct = config.direct
        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path + '_' + config.task_name
        self.vis_path = config.vis_path + '_' + config.task_name
        self.snapshot_path = config.snapshot_path + '_' + config.task_name
        self.result_path = config.vis_path + '_' + config.task_name
        self.tensorboard_path = config.tensorboard_path + '_' + config.task_name

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)
        
        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)

        self.build_model()
        # Start with trained model
        if self.checkpoint:
            self.load_checkpoint()

        #for recording
        self.start_time = time.time()
        self.e = 0
        self.i = 0
        self.loss = {}

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for i in self.cls:
            for param_group in getattr(self, "d_" + i + "_optimizer").param_groups:
                param_group['lr'] = d_lr

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)
    
    def log_tensorboard(self):
        for tag, value in self.loss.items():
            self.writer.add_scalar(tag, value, self.e*self.iters_per_epoch + self.i+1)

    def save_models(self):
        torch.save(self.G.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        for i in self.cls:
            torch.save(getattr(self, "D_" + i).state_dict(),
                       os.path.join(self.snapshot_path, '{}_{}_D_'.format(self.e + 1, self.i + 1) + i + '.pth'))

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def load_checkpoint(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G.pth'.format(self.checkpoint))))
        for i in self.cls:
            getattr(self, "D_" + i).load_state_dict(torch.load(os.path.join(
                self.snapshot_path, '{}_D_'.format(self.checkpoint) + i + '.pth')))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))
    
    def build_model(self):
        # Define generators and discriminators
        self.E = network2.Encoder(self.e_conv_dim)
        self.G = network2.Generator(self.g_conv_dim)
        for i in self.cls:
            setattr(self, "D_" + i, net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm))
        
        # Define vgg for perceptual loss
        self.vgg = net.VGG()
        self.vgg.load_state_dict(torch.load('addings/vgg_conv.pth'))

        # Define loss
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1_none_reduct = torch.nn.L1Loss(reduction='none')
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)

        # Optimizers
        self.e_optimizer = torch.optim.Adam(self.E.parameters(), self.e_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        for i in self.cls:
            setattr(self, "d_" + i + "_optimizer", \
                    torch.optim.Adam(filter(lambda p: p.requires_grad, getattr(self, "D_" + i).parameters()), \
                                     self.d_lr, [self.beta1, self.beta2]))
        
        # Weights initialization
        self.E.apply(self.weights_init_xavier)
        self.G.apply(self.weights_init_xavier)
        for i in self.cls:
            getattr(self, "D_" + i).apply(self.weights_init_xavier)
        
        # Print networks
        self.print_network(self.E, 'E')
        self.print_network(self.G, 'G')
        for i in self.cls:
            self.print_network(getattr(self, "D_" + i), "D_" + i)
        
        if torch.cuda.is_available():
            self.E.cuda()
            self.G.cuda()
            self.vgg.cuda()
            for i in self.cls:
                getattr(self, "D_" + i).cuda()
        
    
    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        index, index_2 = [], []
        for m_A, m_B in zip(mask_A, mask_B):
            index_tmp = m_A.nonzero()
            x_A_index = index_tmp[:, 1]
            y_A_index = index_tmp[:, 2]
            index_tmp = m_B.nonzero()
            x_B_index = index_tmp[:, 1]
            y_B_index = index_tmp[:, 2]
            index.append([x_A_index, y_A_index, x_B_index, y_B_index])
            index_2.append([x_B_index, y_B_index, x_A_index, y_A_index])
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        return mask_A, mask_B, index, index_2
    '''
    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss
    '''
    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255)
        target_data = (self.de_norm(target_data) * 255)
        mask_src = mask_src.expand(-1, 3, mask_src.size(2), mask_src.size(3))
        mask_tar = mask_tar.expand(-1, 3, mask_tar.size(2), mask_tar.size(3))
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = []
        for (i, t, idx) in zip(input_masked, target_masked, index):
            input_match.append(histogram_matching(i, t, idx))
        input_match = torch.stack(input_match)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss
    
    def lambda_preprocess(self, id_A, class_A, id_B, class_B):
        id_A = np.array(id_A).reshape((len(id_A), 1))
        id_B = np.array(id_B).reshape((len(id_B), 1))
        class_A = np.array(class_A).reshape((len(class_A), 1))
        class_B = np.array(class_B).reshape((len(class_B), 1))

        _id = np.array(id_A == id_B)
        _class = np.array(class_A == class_B)
        
        # class 1 - face same, makeup same
        class1 = _id * _class
        # class 2 - face same, makeup diff
        class2 = _id * (_class == False)
        # class 3 - face diff, makeup diff
        class3 = (_id == False) * (_class == False)

        # class 1,2 일 경우 face loss 계산
        lambda_face = class1 + class2 # _id

        # class 2 일 경우 makeup loss 계산
        lambda_makeup = class2

        # class 2,3 일 경우 나머지 loss 계산
        lambda_rest = class2 + class3 # _class == False

        lambda_face = torch.from_numpy(lambda_face.astype(int)).cuda()
        lambda_makeup = torch.from_numpy(lambda_makeup.astype(int)).cuda()
        lambda_rest = torch.from_numpy(lambda_rest.astype(int)).cuda()

        return lambda_face, lambda_makeup, lambda_rest
        
    def train(self):
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)

        # Start with trained model if exists
        cls_A = self.cls[0]
        cls_B = self.cls[1]
        e_lr = self.e_lr
        g_lr = self.g_lr
        d_lr = self.d_lr

        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
            self.vis_test()
        else:
            start = 0
        
        # Start training
        self.start_time = time.time()
        for self.e in range(start, self.num_epochs):
            for self.i, sampled_batch in enumerate(self.data_loader_train):
                img_A = sampled_batch['image_A']
                img_B = sampled_batch['image_B']

                mask_A = sampled_batch['mask_A']
                mask_B = sampled_batch['mask_B']

                id_A = sampled_batch['id_A']
                id_B = sampled_batch['id_B']

                class_A = sampled_batch['class_A']
                class_B = sampled_batch['class_B']

                lambda_face, lambda_makeup, lambda_rest = self.lambda_preprocess(id_A, class_A, id_B, class_B)

                # Convert tensor to variable
                # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose 
                # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                if self.checkpoint or self.direct:
                    if self.lips==True:
                        mask_A_lip = (mask_A==7).float() + (mask_A==9).float()
                        mask_B_lip = (mask_B==7).float() + (mask_B==9).float()
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                    if self.skin==True:
                        mask_A_skin = (mask_A==1).float() + (mask_A==6).float() + (mask_A==13).float()
                        mask_B_skin = (mask_B==1).float() + (mask_B==6).float() + (mask_B==13).float()
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                    if self.eye==True:
                        mask_A_eye_left = (mask_A==4).float()
                        mask_A_eye_right = (mask_A==5).float()
                        mask_B_eye_left = (mask_B==4).float()
                        mask_B_eye_right = (mask_B==5).float()
                        mask_A_face = (mask_A==1).float() + (mask_A==6).float()
                        mask_B_face = (mask_B==1).float() + (mask_B==6).float()
                        # avoid the situation that images with eye closed
                        if not ((mask_A_eye_left>0).any() and (mask_B_eye_left>0).any() and \
                            (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
                            continue
                        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
                        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
                        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
                            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
                        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
                            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)
                
                
                # ================== Train D ================== #
                # Real
                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)
                # Fake
                pose_A, face_A, makeup_A = self.E(org_A)
                pose_B, face_B, makeup_B = self.E(ref_B)

                fake_A = self.G(pose_A, face_A, makeup_B)
                fake_B = self.G(pose_B, face_B, makeup_A)

                fake_A = Variable(fake_A.data).detach()
                fake_B = Variable(fake_B.data).detach()

                # training D_A, D_A aims to distinguish class B
                out = getattr(self, "D_" + cls_A)(ref_B)
                d_A_loss_real = self.criterionGAN(out, True) # todo - change it to batch wise

                out = getattr(self, "D_" + cls_A)(fake_A)
                d_A_loss_fake = self.criterionGAN(out, False) # todo - change it to batch wise

                # Backward + Optimize
                d_A_loss = (d_A_loss_real + d_A_loss_fake) * 0.5
                getattr(self, "d_" + cls_A + "_optimizer").zero_grad()
                d_A_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_A + "_optimizer").step()

                # Logging
                self.loss = {}
                self.loss['D-A-loss'] = (d_A_loss_real.item() + d_A_loss_fake.item()) * 0.5

                # training D_B, D_B aims to distinguish class A
                out = getattr(self, "D_" + cls_B)(org_A)
                d_B_loss_real = self.criterionGAN(out, True) # todo - change it to batch wise

                out = getattr(self, "D_" + cls_B)(fake_B)
                d_B_loss_fake =  self.criterionGAN(out, False) # todo - change it to batch wise
 
                # Backward + Optimize
                d_B_loss = (d_B_loss_real + d_B_loss_fake) * 0.5
                getattr(self, "d_" + cls_B + "_optimizer").zero_grad()
                d_B_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_B + "_optimizer").step()

                # Logging
                self.loss['D-B-loss'] = (d_B_loss_real.item() + d_B_loss_fake.item()) * 0.5

                 # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    pose_A, face_A, makeup_A = self.E(org_A)
                    pose_B, face_B, makeup_B = self.E(ref_B)

                    # face loss
                    loss_face = self.criterionL1_none_reduct(face_A, face_B)
                    loss_face = loss_face.view(self.batch_size, -1).mean(1, keepdim=True)
                    loss_face = loss_face * lambda_face
                    loss_face = loss_face.mean()
                    
                    # makeup loss
                    loss_makeup = self.criterionL1_none_reduct(makeup_A, makeup_B)
                    loss_makeup = loss_makeup.view(self.batch_size, -1).mean(1, keepdim=True)
                    loss_makeup = loss_makeup * lambda_makeup
                    loss_makeup = loss_makeup.mean()
                    
                    # todo - add logs of makeup loss and face loss
                    # todo - apply makeup loss and face loss to network like under code
                    # self.e_optimizer.zero_grad()
                    # self.g_optimizer.zero_grad()
                    # g_loss.backward(retain_graph=True)
                    # self.e_optimizer.step()
                    # self.g_optimizer.step()

                    # identity loss
                    if self.lambda_idt > 0:
                        # G should be identity if ref_B or org_A is fed
                        idt_A = self.G(pose_A, face_A, makeup_A)
                        idt_B = self.G(pose_B, face_B, makeup_B)
                        
                        loss_idt_A = self.criterionL1_none_reduct(idt_A, org_A)
                        loss_idt_A = loss_idt_A.view(self.batch_size, -1).mean(1, keepdim=True)
                        loss_idt_A = loss_idt_A * lambda_rest
                        loss_idt_A = loss_idt_A.mean()

                        loss_idt_B = self.criterionL1_none_reduct(idt_B, org_B)
                        loss_idt_B = loss_idt_B.view(self.batch_size, -1).mean(1, keepdim=True)
                        loss_idt_B = loss_idt_B * lambda_rest
                        loss_idt_B = loss_idt_B.mean()

                        loss_idt = (loss_idt_A + loss_idt_B) * 10 * self.lambda_idt
                    else:
                        loss_idt = 0
                
                    # Fake
                    fake_A = self.G(pose_A, face_A, makeup_B)
                    fake_B = self.G(pose_B, face_B, makeup_A)

                    # GAN loss D_A(G_A(A))
                    pred_fake = getattr(self, "D_" + cls_A)(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True) # todo - change it to batch wise

                    # GAN loss D_B(G_B(B))
                    pred_fake = getattr(self, "D_" + cls_B)(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True) # todo - change it to batch wise

                    # color_histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    if self.checkpoint or self.direct:
                        if self.lips==True:
                            g_A_lip_loss_his = self.criterionHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip # todo - change it to batch wise
                            g_B_lip_loss_his = self.criterionHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip # todo - change it to batch wise
                            g_A_loss_his += g_A_lip_loss_his
                            g_B_loss_his += g_B_lip_loss_his
                        if self.skin==True:
                            g_A_skin_loss_his = self.criterionHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1 # todo - change it to batch wise
                            g_B_skin_loss_his = self.criterionHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2 # todo - change it to batch wise
                            g_A_loss_his += g_A_skin_loss_his
                            g_B_loss_his += g_B_skin_loss_his
                        if self.eye==True:
                            g_A_eye_left_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye # todo - change it to batch wise
                            g_B_eye_left_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye # todo - change it to batch wise
                            g_A_eye_right_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye # todo - change it to batch wise
                            g_B_eye_right_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye # todo - change it to batch wise
                            g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                            g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his
                    
                    # cycle loss
                    pose_fake_A, face_fake_A, makeup_fake_A = self.E(fake_A)
                    pose_fake_B, face_fake_B, makeup_fake_B = self.E(fake_B)

                    rec_A = self.G(pose_fake_A, face_fake_A, makeup_fake_B)
                    rec_B = self.G(pose_fake_B, face_fake_B, makeup_fake_A)

                    g_loss_rec_A = self.criterionL1(rec_A, org_A) * 10 # self.lambda_A # todo - change it to batch wise
                    g_loss_rec_B = self.criterionL1(rec_B, ref_B) * 10 # self.lambda_B # todo - change it to batch wise

                    # vgg loss
                    vgg_org = self.vgg(org_A, self.content_layer)[0]
                    vgg_org = Variable(vgg_org.data).detach()
                    vgg_fake_A = self.vgg(fake_A, self.content_layer)[0]
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * 10 * self.lambda_vgg # * self.lambda_A * self.lambda_vgg # todo - change it to batch wise
                    
                    vgg_ref = self.vgg(ref_B, self.content_layer)[0]
                    vgg_ref = Variable(vgg_ref.data).detach()
                    vgg_fake_B = self.vgg(fake_B, self.content_layer)[0]
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * 10 * self.lambda_vgg # * self.lambda_B * self.lambda_vgg # todo - change it to batch wise
					
                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt # todo - change it to batch wise
                    if self.checkpoint or self.direct:
                        g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his # todo - change it to batch wise
                    
                    self.e_optimizer.zero_grad()
                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.e_optimizer.step()
                    self.g_optimizer.step()

                    # Logging
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-B-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-loss-org'] = g_loss_rec_A.item()
                    self.loss['G-loss-ref'] = g_loss_rec_B.item()
                    self.loss['G-loss-idt'] = loss_idt.item()
                    self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).item()
                    self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).item()
                    if self.direct:
                        self.loss['G-A-loss-his'] = g_A_loss_his.item()
                        self.loss['G-B-loss-his'] = g_B_loss_his.item()
                
                # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()
                    self.log_tensorboard()

                #plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                #save the images
                if (self.i + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([org_A, ref_B, fake_A, fake_B, rec_A, rec_B])

                # Save model checkpoints
                if (self.i + 1) % self.snapshot_step == 0:
                    self.save_models()

                if (self.i % 100 == 99):
                    plot_fig.flush(self.task_name)

                plot_fig.tick()
            
            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

            if self.e % 2 == 0:
                print("Saving output...")
                self.vis_test()

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path, mode)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)
        self.writer.add_image('Train_Image', self.de_norm(img_train_list.data.squeeze()), self.e*self.iters_per_epoch + self.i+1) # todo - change it to batch wise

    def vis_test(self): # to do - change network  
        # saving test results
        mode = "test_vis"
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            base_A, makeup_A = self.E(real_org)
            base_B, makeup_B = self.E(real_ref)

            fake_A = self.G(base_A, makeup_B)
            fake_B = self.G(base_B, makeup_A)

            base_fake_A, makeup_fake_A = self.E(fake_A)
            base_fake_B, makeup_fake_B = self.E(fake_B)

            rec_A = self.G(base_fake_A, makeup_fake_B)
            rec_B = self.G(base_fake_B, makeup_fake_A)

            image_list.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            vis_train_path = os.path.join(self.result_path, mode)
            result_path_now = os.path.join(vis_train_path, "epoch" + str(self.e))
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, normalize=True)
            #print('Translated test images and saved into "{}"..!'.format(save_path))

    def test(self): # to do - change network 
        # Load trained parameters
        G_path = os.path.join(self.snapshot_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        time_total = 0
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            start = time.time()
            real_org = self.to_var(img_A)
            real_ref = self.to_var(img_B)

            image_list = []
            image_list_0 = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            base_A, makeup_A = self.E(real_org)
            base_B, makeup_B = self.E(real_ref)

            fake_A = self.G(base_A, makeup_B)
            fake_B = self.G(base_B, makeup_A)

            base_fake_A, makeup_fake_A = self.E(fake_A)
            base_fake_B, makeup_fake_B = self.E(fake_B)

            rec_A = self.G(base_fake_A, makeup_fake_B)
            rec_B = self.G(base_fake_B, makeup_fake_A)

            time_total += time.time() - start
            image_list.append(fake_A)
            image_list_0.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            image_list_0 = torch.cat(image_list_0, dim=3)

            result_path_now = os.path.join(self.result_path, "multi")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
            result_path_now = os.path.join(self.result_path, "single")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path_0 = os.path.join(result_path_now, '{}_{}_{}_fake_single.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list_0.data), save_path_0, nrow=1, padding=0, normalize=True)
            print('Translated test images and saved into "{}"..!'.format(save_path))
        print("average time : {}".format(time_total/len(self.data_loader_test)))