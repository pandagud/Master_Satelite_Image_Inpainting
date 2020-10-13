#https://github.com/hukkelas/pytorch-frechet-inception-distance#:~:
#text=pre%2Dtrained%20network.-,Fr%C3%A9chet%20Inception%20Distance,detect%20intra%2Dclass%20mode%20collapse.
import torch
import scipy
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
from src.config_default import TrainingConfig
from torchvision.models import inception_v3
from torch.distributions import MultivariateNormal
import seaborn as sns


# This whole implementation, is based on GAN course from coursera, with inspiration drawn from programming assignments

class FIDCalculator():
    def init(self,realImages,fakeImages,generator):
        self.reals = realImages
        self.fakes = fakeImages
        self.gen = generator
        self.device = TrainingConfig.device
        self.mu_fake = 0
        self.mu_real = 0
        self.sigma_fake = 0
        self.sigma_real = 0
    def matrix_sqrt(self,x):
        y = x.cpu().detach().numpy()
        y = scipy.linalg.sqrtm(y)
        return torch.Tensor(y.real, device=x.device)

    def frechet_distance(self,mu_x,mu_y,sigma_x,sigma_y):
        return torch.norm(mu_x - mu_y) + torch.trace(sigma_x + sigma_y - 2 * self.matrix_sqrt(sigma_x @ sigma_y))

    def preprocess(self,img):
        img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
        return img

    import numpy as np
    def get_covariance(self,features):
        return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

    def get_FID_scores(self):
        fake_features_list = []
        real_features_list = []

        self.gen.eval()

        #CHECK OP PÅ DEN HER ift. hvad vi har
        #OBS OBS OBS
        n_samples = 512  # The total number of samples
        batch_size = 4  # Samples per iteration


        inception_model = inception_v3(pretrained=False)
        inception_model.load_state_dict(torch.load("inception_v3_google-1a9a5a14.pth"))
        inception_model.to(self.device)
        inception_model = inception_model.eval()  # Evaluation mode
        inception_model.fc = torch.nn.Identity()

        cur_samples = 0
        Fake_dataloader_Iterations = iter(self.fakes)
        with torch.no_grad():  # You don't need to calculate gradients here, so you do this to save memory
            try:
                for real_example, _ in tqdm(self.reals, total=n_samples // batch_size):  # Go by batch
                    real_samples = real_example
                    real_features = inception_model(real_samples.to(self.device)).detach().to('cpu')  # Move features to CPU
                    real_features_list.append(real_features)

                        #Usikker på om det her dur
                        #https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke
                    fake_samples, target = next(Fake_dataloader_Iterations)
                    fake_samples = self.preprocess(self.gen(fake_samples))
                    fake_features = inception_model(fake_samples.to(self.device)).detach().to('cpu')
                    fake_features_list.append(fake_features)
                    cur_samples += len(real_samples)
                    if cur_samples >= n_samples:
                        break
            except:
                print("Error in loop")

        #Combine all features of loop
        fake_features_all = torch.cat(fake_features_list)
        real_features_all = torch.cat(real_features_list)
        #Get sigma, my
        self.mu_fake = torch.mean(fake_features_all, 0)  # ,True)#,dim=0,keepdim=True)
        self.mu_real = torch.mean(real_features_all, 0)  # ,True)#,dim=0,keepdim=True)
        self.sigma_fake = self.get_covariance(fake_features_all)
        self.sigma_real = self.get_covariance(real_features_all)

        with torch.no_grad():
            print(self.frechet_distance(self.mu_real, self.mu_fake, self.sigma_real, self.sigma_fake).item())

        self.visualizeResults()


    def visualizeResults(self):
        indices = [2, 4, 5]
        fake_dist = MultivariateNormal(self.mu_fake[indices], self.sigma_fake[indices][:, indices])
        fake_samples = fake_dist.sample((5000,))
        real_dist = MultivariateNormal(self.mu_real[indices], self.sigma_real[indices][:, indices])
        real_samples = real_dist.sample((5000,))

        import pandas as pd
        df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
        df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
        df_fake["is_real"] = "no"
        df_real["is_real"] = "yes"
        df = pd.concat([df_fake, df_real])
        sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')