"""
Module that implements FTS-GAN
"""
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import re
import math

from ftsgan.base_gan import BaseGAN
from ftsgan.multitcn import MultiTCN
from ftsgan.ganloss import GANLoss
from ftsgan.utils import next_path

from tqdm import tqdm
import time

try: 
    __IPYTHON__
    _in_ipython_session = True
    from IPython.display import clear_output
except NameError:
    _in_ipython_session = False


class FTSGAN(BaseGAN):
    def __init__(
        self,
        test_metrics,
        num_series,
        batch_size=512,
        num_epochs=2000,
        workers=4,
        ngpu=0,
        params=None,
        save_point=None,
        overwrite_save=True,
        save_dir=None,
        verbose=50,
        series_names=None
    ):
        """
        FTS-GAN
        """
        super().__init__(
            "FTS-GAN",
            num_series=num_series,
            test_metrics=test_metrics,
            batch_size=batch_size,
            num_epochs=num_epochs,
            workers=workers,
            ngpu=ngpu,
            params=params,
            verbose=verbose,
            save_point=save_point,
            save_dir=save_dir,
            overwrite_save=overwrite_save,
            series_names=series_names
        )

        # Setup parameters
        self.params = self.setup_params()

        # Set up networks
        self.net_g = MultiTCN(
            self.params['g_in_channel'],
            self.params['g_out_channel'],
            self.params['g_hidden_layers'],
            self.params['g_skip_layer'],
            num_series=self.num_series,
            batch_norm=self.params['g_batch_norm'],
            condition=self.params['g_conditioned'],
            num_labels=self.params['num_labels'],
            # series_condition=self.params['g_series_condition'],
            price_condition=self.params['g_price_condition']).to(self.device)

        self.net_d = MultiTCN(
            self.params['d_in_channel'],
            self.params['d_out_channel'],
            self.params['d_hidden_layers'],
            self.params['d_skip_layer'],
            num_series=self.num_series,
            batch_norm=self.params['d_batch_norm'],
            condition=self.params['d_conditioned'],
            num_labels=self.params['num_labels'],
            # series_condition=self.params['d_series_condition'],
            price_condition=self.params['d_price_condition']).to(self.device)

        # Set up optimizers
        self.optimizer_g = optim.AdamW(
            self.net_g.parameters(),
            self.params['g_optim_lr'],
            self.params['g_optim_betas'],
            self.params['g_optim_eps'])
        self.optimizer_d = optim.AdamW(
            self.net_d.parameters(),
            self.params['d_optim_lr'],
            self.params['d_optim_betas'],
            self.params['d_optim_eps'])

        # Set up scheduler
        if self.params['lr_decay_rate'] is not None:
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_g,
                gamma=self.params['lr_decay_rate'],)
                # last_epoch= self.num_epochs - self.params['scheduler_delay'])
            self.scheduler_d = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_d,
                gamma=self.params['lr_decay_rate'],)
                # last_epoch= self.num_epochs - self.params['scheduler_delay'])

        # Set tensors to store losses
        self.losses = self.setup_losses(['g', 'd_real', 'd_fake', 'reg_real', 'reg_fake', 'objective'])

        # Set tensor to store aux_losses
        self.aux_losses = self.setup_aux_losses()

        # Set regularises
        self.r1_gamma = 0.0
        self.r2_gamma = 0.0

        # Save for optimal model
        self.optimal_path = None

        self.label_cond = False

    def setup_params(self):
        setup_params = {
            'g_in_channel': self.num_series * 3,
            'g_out_channel': 1,
            'g_hidden_layers': [50] * 6,
            'g_skip_layer': 50,
            'g_batch_norm': True,
            'g_conditioned': [True] * self.num_series,
            # 'g_series_condition': None,
            'g_price_condition': None,
            'g_loss_function': 'standard',

            'g_optim_lr': 1e-4,
            'g_optim_betas': (0., 0.9),
            'g_optim_eps': 1e-08,

            'd_in_channel': 1,
            'd_out_channel': 1,
            'd_hidden_layers': [50] * 6,
            'd_skip_layer': 50,
            'd_batch_norm': False,
            'd_conditioned': [True] * self.num_series,
            # 'd_series_condition': None,
            'd_price_condition': None,
            'd_loss_function': 'standard',

            'd_optim_lr': 3e-4,
            'd_optim_betas': (0., 0.9),
            'd_optim_eps': 1e-08,

            'lr_decay_rate': None,
            'scheduler_delay': 500,

            'topk': None,
            'objective_features': None,
            'num_labels': 0,

        }
        if self.params is not None:
            setup_params.update(self.params)
        return setup_params

    @staticmethod
    def dict_to_device(dict, device):
        if dict is not None:
            for key, value in dict.items():
                dict[key] = value.to(device)

    def train(self, dataset, optimal_params='early_stopping'):

        print("Starting Training Loop...")
        time.sleep(0.2)
        self.optimal_objecive = 1000
        if optimal_params == 'save':
            self.optimal_path = self.get_path()

        for epoch in tqdm(range(self.num_epochs)):
            self.epoch = epoch

            data, labels = dataset.sample(self.batch_size)
            data = data.to(self.device)
            self.dict_to_device(labels, self.device)
            
            if data.shape[2] != self.num_series:
                raise Exception('Data dimensions does not match specified number of series')

            noise = torch.randn(
                self.batch_size,
                self.params['g_in_channel'],
                self.num_series,
                self.net_g.receptive_field_size() + self.net_d.receptive_field_size() - 1,
                device=self.device
            )
            # print(noise.shape, noise)

            self.step_D(data, noise, labels)

            noise = torch.randn(
                self.batch_size,
                self.params['g_in_channel'],
                self.num_series,
                self.net_g.receptive_field_size() + self.net_d.receptive_field_size() - 1,
                device=self.device
            )

            success = self.step_G(noise, labels)

            if optimal_params == 'early_stopping':
                if success:
                    print("Finished Training")
                    break
            elif optimal_params == 'save':
                if self.objective() < self.optimal_objecive:
                    self.save(self.optimal_path)
                    self.optimal_objecive = self.objective()


            if self.params['lr_decay_rate'] is not None:
                if self.epoch > self.params['scheduler_delay']:
                    self.scheduler_g.step()
                    self.scheduler_d.step()

            if self.verbose > 0 and (epoch + 1) % self.verbose == 0:
                if _in_ipython_session:
                    clear_output(wait=True)
                else:
                    os.system('cls' if os.name == 'nt' else 'clear')
                self.output_aux_losses()

            if self.save_point is not None and (epoch + 1) % self.savepoint == 0:
                self.save()

    def generate(self, noise, labels=None, optimal_params='early_stopping'):
        if optimal_params == 'save':
            self.load(self.optimal_path)
            print('epoch optimal:', self.epoch)
        return self.net_g(noise, labels)

    def objective(self):
        return self.losses['objective'][self.epoch].item()

    def inverse_transform(self, assets, dataset, skip_transforms=None):
        synthetic_assets = {}
        synthetic_paths = {}
        for key, value in assets.items():
            print(key)
            if skip_transforms is None:
                skip = ()
            elif key not in skip_transforms:
                skip = ()
            else:
                skip = skip_transforms[key]
            synthetic_assets[key] = dataset.series[key].pipeline.inverse_transform(value, skip)
            synthetic_paths[key] = synthetic_assets[key] * dataset.series[key].raw_data[0, 0]

        return synthetic_assets, synthetic_paths

    @staticmethod
    def scale(data, dim=(0, 2)):
        return (data - data.mean(dim=dim, keepdims=True)) / (data.std(dim=dim, keepdims=True))

    def generate_inverse(self, num_samples, t_dim, dataset, labels=None, optimal_params='early_stopping', pre_transform_dims=None, skip_transforms=None, accept_reject=False, gamma_quartile=0.8, epsilon=1e-12):
        with torch.no_grad():
            if accept_reject:
                synthetic_data = self.accept_reject(num_samples * 4, num_samples, t_dim, gamma_quartile, epsilon)
            else:
                noise = torch.randn(num_samples, self.params['g_in_channel'], self.num_series, t_dim, device=self.device)
                synthetic_data = self.generate(noise, labels, optimal_params)
            # synthetic_data = (synthetic_data - synthetic_data.mean(dim=(3), keepdims=True)) / (synthetic_data.std(dim=(3), keepdims=True))
        synthetic_returns = {}
        for idx, series in enumerate([*dataset.series]):
            synthetic_returns[series] = synthetic_data[:, :, idx, :]
            if pre_transform_dims is not None:
                if series in pre_transform_dims:
                    synthetic_returns[series] = self.scale(synthetic_returns[series], dim=pre_transform_dims[series])
            else:
                synthetic_returns[series] = self.scale(synthetic_returns[series])
        synthetic_assets, synthetic_paths = self.inverse_transform(synthetic_returns, dataset, skip_transforms=skip_transforms)
        return synthetic_returns, synthetic_assets, synthetic_paths

    def step_G(self, noise, labels):
        self.toggle_grad(self.net_g, True)
        self.toggle_grad(self.net_d, False)
        self.optimizer_g.zero_grad()
        fake = self.net_g(noise, labels)
        output = self.net_d(fake, labels)

        loss = GANLoss(topk=self.params['topk'])
        g_loss = loss(output, True)

        aux_loss_log = {}
        overall_success = True
        self.objective_count = 0
        for series in range(self.num_series):
            aux_loss_log[series] = self.compute_test_metrics(series, fake[:, -1:, series, :])
            success = all([aux_loss.success.item() for aux_loss in self.test_metrics[series]])
            overall_success = all([overall_success] + [success])

        if 'all' in self.test_metrics:
            aux_loss_log['all'] = self.compute_test_metrics('all', fake[:, -1:, 0, :], fake_y=fake[:, -1:, 1, :])
            success = all([aux_loss.success.item() for aux_loss in self.test_metrics['all']])
            overall_success = all([overall_success] + [success])

        if 'channel_spread' in self.test_metrics:
            aux_loss_log['channel_spread'] = self.compute_test_metrics('channel_spread', fake[:, :, 0, :], fake_y=fake[:, :, 1, :])
            success = all([aux_loss.success.item() for aux_loss in self.test_metrics['channel_spread']])
            overall_success = all([overall_success] + [success])           

        if not overall_success:
            g_loss.backward()
            self.optimizer_g.step()
        # Aux lossses
        self.losses['objective'][self.epoch] /= self.objective_count
        for series in range(self.num_series):
            for name, loss in aux_loss_log[series]:
                self.aux_losses[series][name][self.epoch] = loss.item()
        if 'all' in self.test_metrics:
            for name, loss in aux_loss_log['all']:
                self.aux_losses['all'][name][self.epoch] = loss.item()
        if 'channel_spread' in self.test_metrics:
            for name, loss in aux_loss_log['channel_spread']:
                self.aux_losses['channel_spread'][name][self.epoch] = loss.item()

        self.losses['g'][self.epoch] = g_loss.item()
        return overall_success

    def step_D(self, x_real, noise, labels):
        self.toggle_grad(self.net_g, False)
        self.toggle_grad(self.net_d, True)
        self.optimizer_d.zero_grad()

        # Real Data
        x_real.requires_grad_()
        d_real = self.net_d(x_real, labels)
        loss = GANLoss()
        d_loss_real = loss(d_real, True)

        # Fake Data
        with torch.no_grad():
            x_fake = self.net_g(noise)

        x_fake.requires_grad_()
        d_fake = self.net_d(x_fake)
        d_loss_fake = loss(d_fake, False)

        # Compute regularizers
        reg_real, reg_fake = self.compute_regularizer(d_real, d_fake, x_real, x_fake)
        d_loss_real.backward()
        d_loss_fake.backward()
        self.optimizer_d.step()
        # Toggle gradients
        self.toggle_grad(self.net_d, False)

        self.losses['d_real'][self.epoch] = d_loss_real.item()
        self.losses['d_fake'][self.epoch] = d_loss_fake.item()
        self.losses['reg_real'][self.epoch] = reg_real.item()
        self.losses['reg_fake'][self.epoch] = reg_fake.item()

    def save(self, path=None):
        if path is None:
            path = self.get_path()
        torch.save({
            'epoch': self.epoch,
            'discriminator_state_dict': self.net_d.state_dict(),
            'generator_state_dict': self.net_g.state_dict(),
            'discriminator_optimizer_state_dict': self.optimizer_d.state_dict(),
            'generator_optimizer_state_dict': self.optimizer_g.state_dict(),
            'losses': self.losses,
            'aux_losses': self.aux_losses
        }, path)

    def load(self, path):
        print(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.net_d.load_state_dict(checkpoint['discriminator_state_dict'])
        self.net_g.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.losses = checkpoint['losses']
        self.aux_losses = checkpoint['aux_losses']
        self.epoch = checkpoint['epoch']

    def get_path(self):
        if self.save_dir is None and self.overwrite_save:
            wkdir = os.getcwd()
            path = wkdir + '/' + self.NAME + '.pt'
        elif self.save_dir is None and not self.overwrite_save:
            wkdir = os.getcwd()
            check_path = wkdir + '/' + self.NAME + '.pt'
            if os.path.isfile(check_path):
                path_pattern = self.save_dir + '/' + self.Name + '-%s.pt'
                path = next_path(path_pattern)
            else:
                path = check_path
        elif self.save_dir is not None and self.overwrite_save:
            path = self.save_dir + '/' + self.NAME + '.pt'
        elif self.save_dir is not None and not self.overwrite_save:
            check_path = self.save_dir + '/' + self.NAME + '.pt'
            if os.path.isfile(check_path):
                path_pattern = self.save_dir + '/' + self.Name + '-%s.pt'
                path = next_path(path_pattern)
            else:
                path = check_path
        return path

    def output_aux_losses(self):
        line = "{:<20}".format("")
        for series in range(self.num_series):
            line += "{:<20}".format(series)
        print(line)

        for key, _ in self.aux_losses[0].items():
            line = '{:<20}'.format(key + ':')
            for series in range(self.num_series):
                line += '{:<20.4f}'.format(self.aux_losses[series][key][self.epoch].item())
            print(line)
        if 'all' in self.test_metrics:
            for key, value in self.aux_losses['all'].items():
                line = '{:<20}'.format(key + ':')
                line += '{:<20.4f}'.format(value[self.epoch].item())
                print(line)
        if 'channel_spread' in self.test_metrics:
            for key, value in self.aux_losses['channel_spread'].items():
                line = '{:<20}'.format(key + ':')
                line += '{:<20.4f}'.format(value[self.epoch].item())
                print(line)

    def accept_reject(self, burin_samples, total_samples, t_dim, gamma_quartile, epsilon):
        max_M = [0] * 2
        max_logit = [0] * 2
        processed_samples = 0

        while processed_samples < burin_samples:
            noise = torch.randn(self.batch_size, self.params['g_in_channel'], self.num_series, t_dim, device=self.device)
            output = self.net_g(noise)
            discrim_logits = self.net_d(output)
            discrim_logits = torch.mean(discrim_logits, dim=3, keepdims=True).squeeze()
            num_series = discrim_logits.shape[1]

            batch_ratio = torch.exp(discrim_logits)
            max_idx = torch.argmax(batch_ratio, dim=0)
            max_ratio = []
            for series in range(num_series):
                max_ratio = batch_ratio[max_idx[series], series]
                if max_ratio > max_M[series]:
                    max_M[series] = max_ratio
                    max_logit[series] = discrim_logits[max_idx[series], series]

            processed_samples += self.batch_size

        counter = 0
        rejected_counter = 0
        p_bar = tqdm(total=total_samples)
        saved_samples = []
        while counter < total_samples:
            noise = torch.randn(self.batch_size, self.params['g_in_channel'], self.num_series, t_dim, device=self.device)
            output = self.net_g(noise)
            discrim_logits = self.net_d(output)
            discrim_logits = torch.mean(discrim_logits, dim=3, keepdims=True).squeeze()

            batch_ratio = torch.exp(discrim_logits)
            max_idx = torch.argmax(batch_ratio, dim=0)
            max_ratio = []

            acceptance_prob = torch.zeros(num_series, self.batch_size)
            for series in range(num_series):
                max_ratio = batch_ratio[max_idx[series], series]

                if max_ratio > max_M[series]:
                    max_M[series] = max_ratio
                    max_logit[series] = discrim_logits[max_idx[series], series]

                Fs = discrim_logits[:, series] - max_logit[series] - torch.log(1 - torch.exp(discrim_logits[:, series] - max_logit[series] - epsilon))
                gamma = torch.quantile(Fs, gamma_quartile)
                F_hat = Fs - gamma
                acceptance_prob[series] = torch.sigmoid(F_hat)

            for idx, sample in enumerate(output):
                probability = torch.rand(self.num_series)
                compare = torch.ge(acceptance_prob[:, idx], probability)
                if torch.all(compare):
                    saved_samples.append(sample)
                    counter += 1
                    p_bar.update(1)
                    if counter >= total_samples:
                        break

        p_bar.close()

        return torch.cat(saved_samples, dim=0).unsqueeze(1)
