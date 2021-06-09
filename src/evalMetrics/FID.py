import torch
import scipy
import numpy as np
#import seaborn as sns
import pandas as pd
import os
from tqdm.auto import tqdm
from pathlib import Path
from src.config_default import TrainingConfig
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal

#https://github.com/hukkelas/pytorch-frechet-inception-distance#:~:
#text=pre%2Dtrained%20network.-,Fr%C3%A9chet%20Inception%20Distance,detect%20intra%2Dclass%20mode%20collapse.
# This whole implementation, is based on GAN course from coursera
#https://github.com/mseitzer/pytorch-fid
class FIDCalculator:
    def __init__(self,realImages,fakeImages, numberOfSamples,batchsize,config,TCI=None):
        self.reals = realImages
        self.fakes = fakeImages
        #self.gen = generator
        self.device = TrainingConfig.device
        self.numberOfSamples = numberOfSamples
        self.batchsize = batchsize
        self.mu_fake = 0
        self.mu_real = 0
        self.sigma_fake = 0
        self.sigma_real = 0
        self.config =config
        self.TCI = TCI
    def matrix_sqrt(self,x):
        y = x.cpu().detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device=x.device)

    def frechet_distance(self,mu_x,mu_y,sigma_x,sigma_y):
        return torch.norm(mu_x - mu_y) + torch.trace(sigma_x + sigma_y - 2 * self.matrix_sqrt(sigma_x @ sigma_y))

    def preprocess(self,img):
        img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        return img

    def get_covariance(self,features):
        features = features.detach().numpy()
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        cov = np.cov(features,rowvar=False)
        tensor = torch.Tensor(cov)
        return tensor

    def get_FID_scores(self):
        fake_features_list = []
        real_features_list = []

        #self.gen.eval()

        #CHECK OP PÅ DEN HER ift. hvad vi har
        #OBS OBS OBS
        n_samples = self.numberOfSamples  # The total number of samples
        batch_size = self.batchsize  # Samples per iteration


        inception_model = inception_v3(pretrained=False)
        #find pretrained downlaoded model
        if self.config.run_polyaxon:
            finalPath= self.config.data_path / 'models' / 'OutputModels' /'inception_v3_google-1a9a5a14.pth'
        else:
            localdir = Path().absolute().parent
            modelOutputPath = Path.joinpath(localdir, 'OutputModels')
            finalPath = Path.joinpath(modelOutputPath, 'inception_v3_google-1a9a5a14.pth')

        inception_model.load_state_dict(torch.load(finalPath))
        inception_model.to(self.device)
        inception_model = inception_model.eval()  # Evaluation mode
        inception_model.fc = torch.nn.Identity()

        cur_samples = 0
        Fake_dataloader_Iterations = iter(self.fakes)
        with torch.no_grad():  # You don't need to calculate gradients here, so you do this to save memory
            try:
                if self.TCI:
                    for real_example,target in tqdm(self.reals, total=n_samples // batch_size,
                                             disable=self.config.run_polyaxon):  # Go by batch
                        real_samples = real_example
                        real_features = inception_model(real_samples.to(self.device)).detach().to(
                            'cpu')  # Move features to CPU
                        real_features_list.append(real_features)

                        # Usikker på om det her dur
                        # https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke
                        fake_samples = next(Fake_dataloader_Iterations)
                        # fake_samples = self.preprocess(self.gen(fake_samples))
                        fake_features = inception_model(fake_samples[0].to(self.device)).detach().to('cpu')
                        fake_features_list.append(fake_features)
                        cur_samples += len(real_samples)
                        if cur_samples >= n_samples:
                            break
                else :
                    for real_example in tqdm(self.reals, total=n_samples // batch_size,
                                             disable=self.config.run_polyaxon):  # Go by batch
                        real_samples = real_example
                        real_features = inception_model(real_samples.to(self.device)).detach().to(
                            'cpu')  # Move features to CPU
                        real_features_list.append(real_features)

                        # Usikker på om det her dur
                        # https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke
                        fake_samples = next(Fake_dataloader_Iterations)
                        # fake_samples = self.preprocess(self.gen(fake_samples))
                        fake_features = inception_model(fake_samples.to(self.device)).detach().to('cpu')
                        fake_features_list.append(fake_features)
                        cur_samples += len(real_samples)
                        if cur_samples >= n_samples:
                            break
            except:
                print("Error in FID loop")


        #Combine all features of loop
        fake_features_all = torch.cat(fake_features_list)
        real_features_all = torch.cat(real_features_list)
        #Get sigma, my
        self.mu_fake = torch.mean(fake_features_all, 0)  # ,True)#,dim=0,keepdim=True)
        self.mu_real = torch.mean(real_features_all, 0)  # ,True)#,dim=0,keepdim=True)
        self.sigma_fake = self.get_covariance(fake_features_all)
        self.sigma_real = self.get_covariance(real_features_all)

        with torch.no_grad():
            FID_Distance = self.frechet_distance(self.mu_real, self.mu_fake, self.sigma_real, self.sigma_fake).item()
        #If you want to visualize uncomment this
        #self.visualizeResults()
        return FID_Distance

    def visualizeResults(self):
        indices = [2, 4, 5]
        fake_dist = MultivariateNormal(self.mu_fake[indices], self.sigma_fake[indices][:, indices])
        fake_samples = fake_dist.sample((5000,))
        real_dist = MultivariateNormal(self.mu_real[indices], self.sigma_real[indices][:, indices])
        real_samples = real_dist.sample((5000,))

        df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
        df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
        df_fake["is_real"] = "no"
        df_real["is_real"] = "yes"
        df = pd.concat([df_fake, df_real])
        #sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')