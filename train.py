import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import os

from models import *

class Trainer():

    def __init__(self, data_directory, class_size, embedding_dim, batch_size, latent_size=100, device='cpu', lr=0.0002, num_workers=1):
        #load dataset
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

        #check if data directory exists, if not, create it.
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print('Directory created.')
        else:
            print('Directory exists.')

        #get dataset from directory. If not present, download to directory
        self.dataset = torchvision.datasets.MNIST(data_directory, train=True, transform=transformation, download=True)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size

        #define models
        self.latent_size = 100 

        self.dis = Discriminator(class_size, embedding_dim).to(device)
        self.gen = Generator(latent_size,class_size, embedding_dim).to(device)

        self.loss_func = nn.BCELoss().to(device)

        self.optimizer_d = optim.Adam(self.dis.parameters(), lr=lr)
        self.optimizer_g = optim.Adam(self.gen.parameters(), lr=lr)

    def train(self, epochs, saved_image_directory, saved_model_directory):
        start_time = time.time()

        gen_loss_list = []
        dis_loss_list = []

        for epoch in range(epochs):
            gen_loss = 0
            dis_loss = 0
            cur_time = time.time()
            for images, labels in self.data_loader:
                b_size = len(images)
                #train Discriminator
                self.optimizer_d.zero_grad()

                #fake loss
                rand_labels = torch.LongTensor(np.random.randint(0, self.class_size, b_size)).to(self.device)
                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, rand_labels)
                fake_pred = self.dis(fake_images, rand_labels)
                fake_labels = torch.full((b_size,), 0).to(self.device)
                d_loss_fake = self.loss_func(fake_pred, fake_labels)

                #real loss
                real_labels = torch.full((b_size,), 1).to(self.device)
                real_pred = self.dis(images.to(self.device), labels.to(self.device))
                d_loss_real = self.loss_func(real_pred, real_labels)

                d_loss = (d_loss_real + d_loss_fake)*(1/2)
                d_loss.backward()
                self.optimizer_d.step()

                dis_loss = d_loss.item()/b_size

                #train Generator
                self.optimizer_g.zero_grad()

                rand_labels = torch.LongTensor(np.random.randint(0, self.class_size, b_size)).to(self.device)
                z = torch.randn(b_size, self.latent_size).to(self.device)
                fake_images = self.gen(z, rand_labels)
                fake_pred = self.dis(fake_images, rand_labels)
                g_loss = self.loss_func(fake_pred, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                gen_loss = g_loss.item()/b_size

            cur_time = time.time() - cur_time

            print('Epoch {},    Gen Loss: {:.4f},   Dis Loss: {:.4f}'.format(epoch, gen_loss, dis_loss))
            print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, (epochs-epoch)*(cur_time)/3600))
            gen_loss_list.append(gen_loss)
            dis_loss_list.append(dis_loss)

            #show samples
            labels = torch.LongTensor(np.arange(10)).to(self.device)
            z = torch.randn(10, self.latent_size).to(self.device)
            sample_images = self.gen(z, labels)

            #save models to model_directory
            torch.save(self.gen.state_dict(), saved_model_directory + '/generator_{}.pt'.format(epoch))
            torch.save(self.dis.state_dict(), saved_model_directory + '/discriminator_{}.pt'.format(epoch))

            image_grid = torchvision.utils.make_grid(sample_images.cpu().detach(), nrow=5, normalize=True)
            _, plot = plt.subplots(figsize=(12, 12))
            plt.axis('off')
            plot.imshow(image_grid.permute(1, 2, 0))
            plt.savefig(saved_image_directory + '/epoch_{}_checkpoint.jpg'.format(epoch), bbox_inches='tight')

        finish_time = time.time() - start_time
        print('Training Finished. Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
        return gen_loss_list, dis_loss_list

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training GAN')
    #hyperparameter loading
    parser.add_argument('--data_directory', type=str, default='data', help='directory to MNIST dataset files')
    parser.add_argument('--saved_image_directory', type=str, default='data/saved_images', help='directory to where image samples will be saved')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models', help='directory to where model weights will be saved')
    parser.add_argument('--class_size', type=int, default=10, help='number of unique classes in dataset')
    parser.add_argument('--embedding_dim', type=int, default=10, help='size of embedding vector')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batches passed through networks at each step')
    parser.add_argument('--latent_size', type=int, default=100, help='size of gaussian noise vector')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu depending on availability and compatability')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of models')
    parser.add_argument('--num_workers', type=int, default=0, help='workers simultaneously putting data into RAM')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations of dataset through network for training')
    args = parser.parse_args()

    data_dir = args.data_directory
    saved_image_dir = args.saved_image_directory
    saved_model_dir = args.saved_model_directory
    class_size = args.class_size
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    latent_size = args.latent_size
    device = args.device
    lr = args.lr
    num_workers = args.num_workers
    epochs = args.epochs

    gan = Trainer(data_dir, class_size, embedding_dim, batch_size, latent_size, device, lr, num_workers)
    gen_loss_lost, dis_loss_list = gan.train(epochs, saved_image_dir, saved_model_dir)


if __name__ == "__main__":
    main()
