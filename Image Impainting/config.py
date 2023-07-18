import torch
import os

# GENERIC
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INIMAGE_FOLDER = "./input"
MASKIMAGE_FOLDER = "./mask"
INIMAGE = "./input"
MASKIMAGE = "./mask"
#INIMAGE = ['./input/input_img.png', './input/01235.png']
#MASKIMAGE = ['./mask/input_img.png', './mask/01235.png']
OUTIMAGE = "./output"
RESIZE_TO = (512,512)
CUDA = True if torch.cuda.is_available() else False


# DEEPFILLv2
DEEPFILL_MODEL_PATH = "./model/deepfillv2_WGAN.pth"
GPU_ID = -1
INIT_TYPE = "xavier"
INIT_GAIN = 0.02
PAD_TYPE = "zero"
IN_CHANNELS = 4
OUT_CHANNELS = 3
LATENT_CHANNELS = 48
ACTIVATION = "elu"
NORM = "in"
NUM_WORKERS = 0

'''if __name__ == "__main__":
	for filename in os.listdir(INIMAGE_FOLDER):
		f = os.path.join(INIMAGE_FOLDER, filename)
		if(f[len(f)-1]=='g' and f[len(f)-2]=='n' and f[len(f)-3]=='p' and f[len(f)-4]=='.'):
			INIMAGE.append(f)

	for filename in os.listdir(MASKIMAGE_FOLDER):
		f = os.path.join(MASKIMAGE_FOLDER, filename)
		if(f[len(f)-1]=='g' and f[len(f)-2]=='n' and f[len(f)-3]=='p' and f[len(f)-4]=='.'):
			MASKIMAGE.append(f)


	print(INIMAGE)

	print(MASKIMAGE)'''
	
