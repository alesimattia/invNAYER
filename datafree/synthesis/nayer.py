import copy

import datafree
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from torchvision import transforms
from kornia import augmentation
import time


def reptile_grad(src, tar):
	for p, tar_p in zip(src.parameters(), tar.parameters()):
		if p.grad is None:
			p.grad = Variable(torch.zeros(p.size())).cuda()
		p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
	for p, tar_p in zip(src.parameters(), tar.parameters()):
		if p.grad is None:
			p.grad = Variable(torch.zeros(p.size())).cuda()
		p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def reset_l0(model):
	for n, m in model.named_modules():
		if n == "l1.0" or n == "conv_blocks.0":
			nn.init.normal_(m.weight, 0.0, 0.02)
			nn.init.constant_(m.bias, 0)

def reset_g(model):
	for m in model.modules():
		if isinstance(m, (nn.Conv2d)):
			nn.init.xavier_uniform_(m.weight)
		if isinstance(m, (nn.BatchNorm2d)):
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)
		if isinstance(m, (nn.Linear)):
			nn.init.normal_(m.weight, mean=0, std=1)
			nn.init.constant_(m.bias, 0)


def reset_g1(model):
	for m in model.modules():
		if isinstance(m, (nn.Conv2d)):
			nn.init.xavier_uniform_(m.weight)
		if isinstance(m, (nn.Linear)):
			nn.init.normal_(m.weight, mean=0, std=1)
			nn.init.constant_(m.bias, 0)


def reset_bn(model):
	for m in model.modules():
		if isinstance(m, (nn.BatchNorm2d)):
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0)


def custom_cross_entropy(preds, target):
	return torch.mean(torch.sum(-target * preds.log_softmax(dim=-1), dim=-1))



def get_image_prior_losses(inputs_jit):
		# Perdita di variazione totale
		diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
		diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
		diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
		diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

		loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
		loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
				diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
		loss_var_l1 = loss_var_l1 * 255.0
		return loss_var_l1, loss_var_l2


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class NAYER(BaseSynthesis):
	def __init__(self, teacher, student, generator, num_classes, img_size,
				 init_dataset=None, g_steps=100, lr_g=0.1,
				 synthesis_batch_size=128, sample_batch_size=128,
				 adv=0.0, bn=1, oh=1, num_workers=4, contr=1,
				 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
				 normalizer=None, device='cpu', distributed=False,
				 warmup=10, bn_mmt=0, bnt=30, oht=1.5,
				 cr_loop=1, g_life=50, g_loops=1, gwp_loops=10, dataset="cifar10", coefficients=None):
		super(NAYER, self).__init__(teacher, student)

		if "r_feature" in coefficients:
			self.bn_reg_scale = coefficients["r_feature"]
			self.first_bn_multiplier = coefficients["first_bn_multiplier"]
			self.coeff_var_l1 = coefficients["tv_l1"]
			self.coeff_var_l2 = coefficients["tv_l2"]
			self.coeff_l2 = coefficients["l2"]
			self.main_loss_multiplier = coefficients["main_loss_multiplier"]
			self.adi_scale = coefficients["adi_scale"]

			## Create hooks for feature statistics
			self.loss_r_feature_layers = []
			for module in self.teacher.modules():
				if isinstance(module, nn.BatchNorm2d):
					self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))
		else:
			print("Dict. r_feature incompleto")
			
		self.save_dir = save_dir
		self.img_size = img_size
		self.g_steps = g_steps

		self.lr_g = lr_g
		self.adv = adv
		self.bn = bn
		self.oh = oh
		self.contr = contr
		self.bn_mmt = bn_mmt
		self.num_workers = num_workers

		self.num_classes = num_classes
		self.distributed = distributed
		self.synthesis_batch_size = int(synthesis_batch_size/cr_loop)
		self.sample_batch_size = sample_batch_size
		self.init_dataset = init_dataset
		self.use_fp16 = use_fp16
		self.autocast = autocast  # for FP16
		self.normalizer = normalizer
		self.data_pool = ImagePool(root=self.save_dir)
		self.transform = transform
		self.data_iter = None
		self.generator = generator.to(device).train()
		self.device = device
		self.hooks = []

		self.ep = 0
		self.ep_start = warmup

		self.g_life = g_life
		self.bnt = bnt
		self.oht = oht
		self.g_loops = g_loops
		self.gwp_loops = gwp_loops
		self.dataset = dataset
		self.label_list = torch.LongTensor([i for i in range(self.num_classes)])

		for m in teacher.modules():
			if isinstance(m, nn.BatchNorm2d):
				self.hooks.append(DeepInversionHook(m, self.bn_mmt))

		if dataset == "imagenet" or dataset == "tiny_imagenet":
			self.aug = transforms.Compose([
				augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
				normalizer,
			])
		else:
			self.aug = transforms.Compose([
				augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4, p=0.5),
				augmentation.RandomHorizontalFlip(),
				normalizer
			])

	def jitter_and_flip(self, inputs_jit, lim=1. / 8., do_flip=True):
		lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

		# apply random jitter offsets
		off1 = random.randint(-lim_0, lim_0)
		off2 = random.randint(-lim_1, lim_1)
		inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

		# Flipping
		flip = random.random() > 0.5
		if flip and do_flip:
			inputs_jit = torch.flip(inputs_jit, dims=(3,))
		return inputs_jit

	def synthesize(self, targets=None):
		start = time.time()
		self.student.eval()
		self.teacher.eval()
		best_cost = 1e6
		best_oh = 1e6
		best_loss_var_l1= 1e6
		best_loss_var_l2 = 1e6
		best_loss_l2 = 1e6

		if (self.ep - self.ep_start) % self.g_life == 0 or self.ep % self.g_life == 0:
			self.generator = self.generator.reinit()

		if self.ep < self.ep_start:
			g_loops = self.gwp_loops
		else:
			g_loops = self.g_loops
		self.ep += 1
		bi_list = []
		if g_loops == 0:
			return None, 0, 0, 0
		if self.dataset == "imagenet":
			idx = torch.randperm(self.label_list.shape[0])
			self.label_list = self.label_list[idx]
		for gs in range(g_loops):
			best_inputs = None
			self.generator.re_init_le()

			if self.dataset == "imagenet":
				targets, ys = self.generate_ys_in(cr=0.0, i=gs)
				print(targets)
			else:
				targets, ys = self.generate_ys(cr=0.0)
			ys = ys.to(self.device)
			targets = targets.to(self.device)

			optimizer = torch.optim.Adam([
				{'params': self.generator.parameters()},
			], lr=self.lr_g, betas=[0.5, 0.999])

			# Adeguamento per contrastive loss
			for it in range(self.g_steps):
				inputs = self.generator(targets=targets)
				if self.dataset == "imagenet":
					inputs = self.jitter_and_flip(inputs)
					inputs_aug = self.aug(inputs)
				else:
					inputs_aug = self.aug(inputs)

				if(self.contr != 0 ): #Contrastive loss active
					t_out = self.teacher(inputs)
					t_out_aug = self.teacher(inputs_aug)
					differences = torch.argmax(t_out, dim=1) != torch.argmax(t_out_aug, dim=1)
					if torch.sum(differences).item() == 0:
						t_out = t_out_aug
					else:
						# I want G to synthesize samples that produce the same response on T regardless of the augmentation
						# TODO: can semantics (inter-class similarities) be useful here?
						# TODO: check if self.aug is used during T training. the end-user cannot know what augmentations were used   
						# add a loss term that penalizes G in this case
						num_differences = torch.sum(differences) # .item()
						loss_contr = num_differences / t_out.size()[0]
				else: #Standard Nayer
					loss_contr = 0
					t_out = self.teacher(inputs_aug)

				################### MODIFICA FUNZIONE DI LOSS #######################
				loss_oh = custom_cross_entropy(t_out, ys.detach())
				
				if self.adv > 0 and (self.ep > self.ep_start):
					s_out = self.student(inputs_aug)
					mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
					loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
						1) * mask).mean()  # decision adversarial distillation
				else:
					loss_adv = loss_oh.new_zeros(1)

				''' 
					Calcolo loss_r_feature come in DeepInversion 
					Elimina loss_bn
				'''
				rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.hooks)-1)]
				loss_bn = sum([h.r_feature for h in self.hooks])

				######### CALCOLO COMPONENTE "loss_aux" ########
				loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_aug) # R_prior
				loss_l2 = torch.norm(inputs_aug.view(inputs_aug.size(0), -1), dim=1).mean()

				loss_prior = 	self.coeff_var_l2 * loss_var_l2 + \
								self.coeff_var_l1 * loss_var_l1 + \
								self.coeff_l2 * loss_l2
				loss_r_feature = self.bn_reg_scale * (sum([h.r_feature * rescale[idx] for idx, h in enumerate(self.hooks)]))
				#########################################################
				loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv \
						+ loss_prior + loss_r_feature


				if loss_oh.item() < best_oh:
					best_oh = loss_oh
				if loss_var_l1.item() < best_loss_var_l1:
					best_loss_var_l1 = loss_var_l1
				if loss_var_l2.item() < best_loss_var_l2:
					best_loss_var_l2 = loss_var_l2
				if loss_l2.item() < best_loss_l2:
					best_loss_l2 = loss_l2

				# print("%s - bn %s - bn %s - oh %s - adv %s" % (
				# it, (loss_bn * self.bn).data, loss_bn.data, (loss_oh).data, (self.adv * loss_adv).data))
				print("%s - bn %s - oh %s - adv %s - var_l1 %s - var_l2 %s - l2 %s" % (
				it, (loss_bn * self.bn).data, (loss_oh * self.oh).data, (self.adv * loss_adv).data, 
				(self.coeff_var_l1 * loss_var_l1).data, (self.coeff_var_l2 * loss_var_l2).data, (self.coeff_l2 * loss_l2).data))

				with torch.no_grad():
					if best_cost > loss.item() or best_inputs is None:
						best_cost = loss.item()
						best_inputs = inputs.data

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if self.bn_mmt != 0:
				for h in self.hooks:
					h.update_mmt()

			self.student.train()
			end = time.time()

			self.data_pool.add(best_inputs)
			bi_list.append(best_inputs)

			dst = self.data_pool.get_dataset(transform=self.transform)
			if self.init_dataset is not None:
				init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
				dst = torch.utils.data.ConcatDataset([dst, init_dst])
			if self.distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
			else:
				train_sampler = None
			loader = torch.utils.data.DataLoader(
				dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
				num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)
			self.data_iter = DataIter(loader)																			#Per il logger WANDB
		return {"synthetic": bi_list}, end - start, best_cost, best_oh, best_loss_var_l1, best_loss_var_l2, best_loss_l2, loss_bn, loss_adv

	def sample(self):
		return self.data_iter.next()

	def generate_ys_in(self, cr=0.0, i=0):
		target = self.label_list[i*self.synthesis_batch_size:(i+1)*self.synthesis_batch_size]
		target = torch.tensor([250, 230, 283, 282, 726, 895, 554, 555, 105, 107])


		ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
		ys.fill_(cr / (self.num_classes - 1))
		ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

		return target, ys

	def generate_ys(self, cr=0.0):
		s = self.synthesis_batch_size // self.num_classes
		v = self.synthesis_batch_size % self.num_classes
		target = torch.randint(self.num_classes, (v,))
		for i in range(s):
			tmp_label = torch.tensor(range(0, self.num_classes))
			target = torch.cat((tmp_label, target))

		ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
		ys.fill_(cr / (self.num_classes - 1))
		ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))
		print(target)

		return target, ys

	def generate_lys(self, cr=0.0, value=3):
		s = self.synthesis_batch_size // self.num_classes
		v = self.synthesis_batch_size % self.num_classes
		target = torch.randint(self.num_classes, (v,))
		for i in range(s):
			tmp_label = torch.tensor(range(0, self.num_classes))
			target = torch.cat((tmp_label, target))

		yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
		yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))
		yf = yf.to(device=self.device)

		yl = torch.ones(self.synthesis_batch_size, self.num_classes)*(-value)
		yl.scatter_(1, target.data.unsqueeze(1), value)
		yl = yl.to(device=self.device)

		cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

		return target, yf, yl, cr_vec


	def generate_lys_v2(self, cr=0.0, value=3, norm=50):
		s = self.synthesis_batch_size // self.num_classes
		v = self.synthesis_batch_size % self.num_classes
		target = torch.randint(self.num_classes, (v,))
		crate = random.randint(0, int(cr*norm))/norm
		for i in range(s):
			tmp_label = torch.tensor(range(0, self.num_classes))
			target = torch.cat((tmp_label, target))

		yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
		yf.scatter_(1, target.data.unsqueeze(1), (1 - crate))
		yf = yf.to(device=self.device)

		yl = torch.ones(self.synthesis_batch_size, self.num_classes)*(-value)
		yl.scatter_(1, target.data.unsqueeze(1), value)
		yl = yl.to(device=self.device)

		cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*crate

		return target, yf, yl, cr_vec


	def generate_lys_v3(self, cr=0.0):
		s = self.synthesis_batch_size // self.num_classes
		v = self.synthesis_batch_size % self.num_classes
		target = torch.randint(self.num_classes, (v,))
		for i in range(s):
			tmp_label = torch.tensor(range(0, self.num_classes))
			target = torch.cat((tmp_label, target))

		yf = torch.zeros(self.synthesis_batch_size, self.num_classes)
		yf.scatter_(1, target.data.unsqueeze(1), (1 - cr))

		yf = yf.to(device=self.device)

		yl = torch.zeros(size=(self.synthesis_batch_size, self.num_classes), device=self.device)
		cr_vec = torch.ones(size=(self.synthesis_batch_size, self.num_classes), device=self.device)*cr

		return target, yf, yl, cr_vec