import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import Net
#from metrics import get_psnr_ssim
import numpy as np
from torchvision import transforms
#from CR import  *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	dehaze_net = nn.DataParallel(Net.DehazeNet().cuda())
	dehaze_net.apply(weights_init)
	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path)		
	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

	#criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	dehaze_net.train()

	for epoch in range(config.num_epochs):
		for iteration, (img_orig, img_haze) in enumerate(train_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean_image = dehaze_net(img_haze)

			criterion = nn.MSELoss().cuda()
			loss = criterion(clean_image, img_orig)


			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)  # 梯度裁剪原理：既然在BP过程中会产生梯度消失（就是偏导会无线接近0，导致长时记忆无法更新），																				  # 那么最简单最粗暴的方法，就是设定阈值，当梯度小于阈值时，更新的梯度为阈值。
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
			#model_path = "H:/code/comparison algorithm/GCA/snapshots/GCANet/Epoch{}.pth".format(epoch)
			#Avarage_SSIM,Avarage_PSNR = get_psnrs_ssims(dir,model_path)
			#print("[epoch:%2d/%d][PSNR : %.7f SSIM : %.7f ]" % (1 + epoch, config.num_epochs, Avarage_PSNR,Avarage_SSIM))

		# Validation Stage
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):


			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean_image = dehaze_net(img_haze)
			#psnr, ssim = get_psnr_ssim(img_orig, clean_image)
			#print(psnr,ssim)


			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

		torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")

			

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default=r'/data/fwc2/datasets/datasets/Train_data/rain/label/')
	parser.add_argument('--hazy_images_path', type=str, default=r'/data/fwc2/datasets/datasets/Train_data/rain/rain/')
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=16)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")


	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)






