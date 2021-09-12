"""
Defines the base class for GANs
"""
import torch

from ftsgan.evaluation import CorrelationLoss, SpreadLoss


class BaseGAN():
    def __init__(
        self,
        name,
        num_series,
        test_metrics,
        batch_size,
        num_epochs,
        workers,
        ngpu,
        params,
        save_point,
        overwrite_save,
        save_dir,
        verbose,
        series_names=None,
    ):

        # Set number of series
        self.num_series = num_series

        # Series names
        self.series_names = series_names if series_names is not None else range(self.num_series)

        # Setup params
        self.params = params

        # Test metrics for evaluating GAN
        self.test_metrics = test_metrics

        # Set number of gpu
        self.ngpu = ngpu

        # Set device to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        # Set name for the algorithm
        self.NAME = name

        # Set batch size for training
        self.batch_size = batch_size

        # Number of workers for dataloader
        self.workers = workers

        # Set number of epochs to run
        self.num_epochs = num_epochs

        # Set save point
        self.save_point = save_point

        # Overwrite saved file without loading
        self.overwrite_save = overwrite_save

        # Directory to save models
        self.save_dir = save_dir

        # Set whether the model should output losses
        self.verbose = verbose

        # Epoch
        self.epoch = 0

    def compute_test_metrics(self, series, fake, fake_y=None, corr=False):
        test_metrics = list()
        # losses = list()
        for aux_loss in self.test_metrics[series]:
            if isinstance(aux_loss, CorrelationLoss):
                with torch.no_grad():
                    loss = aux_loss(fake.permute(0, 2, 1), fake_y.permute(0, 2, 1))
                    loss = loss.cpu().detach().numpy()
                if self.params['objective_features'] is not None:
                    if aux_loss.name in self.params['objective_features']:
                        self.losses['objective'][self.epoch] += loss.item()
                        self.objective_count += 1
                test_metrics.append((aux_loss.name, loss))
            elif isinstance(aux_loss, SpreadLoss):
                with torch.no_grad():
                    loss = aux_loss(fake.permute(0, 2, 1))
                    loss = loss.cpu().detach().numpy()
                if self.params['objective_features'] is not None:
                    if aux_loss.name in self.params['objective_features']:
                        self.losses['objective'][self.epoch] += loss.item()
                        self.objective_count += 1
                test_metrics.append((aux_loss.name, loss))
            else:
                with torch.no_grad():
                    loss = aux_loss(fake.permute(0, 2, 1))
                    loss = loss.cpu().detach().numpy()
                if self.params['objective_features'] is not None:
                    if aux_loss.name in self.params['objective_features']:
                        self.losses['objective'][self.epoch] += loss.item()
                        self.objective_count += 1
                test_metrics.append((aux_loss.name, loss))
        return test_metrics

    def setup_losses(self, losses):
        loss_dict = {}
        for key in losses:
            loss_dict[key] = torch.zeros(self.num_epochs)

        return loss_dict

    def setup_aux_losses(self):
        loss_dict = {}
        for series in range(self.num_series):
            series_dict = {}
            for aux_loss in self.test_metrics[series]:
                if isinstance(aux_loss, CorrelationLoss):
                    pass
                else:
                    series_dict[aux_loss.name] = torch.zeros(self.num_epochs, device=self.device)

            loss_dict[series] = series_dict
        if 'all' in self.test_metrics:
            all_dict = {}
            for aux_loss in self.test_metrics['all']:
                all_dict[aux_loss.name] = torch.zeros(self.num_epochs, device=self.device)
            loss_dict['all'] = all_dict
        if 'channel_spread' in self.test_metrics:
            spread_dict = {}
            for aux_loss in self.test_metrics['channel_spread']:
                spread_dict[aux_loss.name] = torch.zeros(self.num_epochs, device=self.device)
            loss_dict['channel_spread'] = spread_dict

        return loss_dict

    """
    Following methods taken from:
    M. Wiese, R. Knobloch, R.Korn, and P.Kretchmer, "Quant GANs: deep generation of 
    financial time series," Quantitative Finance, Vol 20, no. 9, pp. 1419-1440, 2020,
    arXiv: 1907.06673
    """
    def toggle_grad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    def compute_regularizer(self, d_real, d_fake, x_real, x_fake):
        reg_1 = torch.tensor(0.)
        reg_2 = torch.tensor(0.)
        if self.r1_gamma > 0.0:
            reg_1 = self.r1_gamma * compute_grad2(d_fake, x_fake).mean()
            reg_1.backward(retain_graph=True)
        if self.r2_gamma > 0.0:
            reg_2 = self.r2_gamma * compute_grad2(d_real, x_real).mean()
            reg_2.backward(retain_graph=True)
        return reg_1, reg_2

    def compute_grad2(d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg   


