# FTS-GAN

This package is an implementation of the FTS-GAN framework from [FTS-GAN: Generating Multidimensional Financial Time Series](FTS-GAN%20Paper.pdf). The framework allows the generation of synthetic data by learning an underlying distribution.

## Getting Started
Data can be loaded from a csv using one of the data loaders. If the S&P 500 and VIX is used as an example to load a csv the following is done:

``` python
from ftsgan.ftsgan import load_data

dataset = load_data(
        file_dir=data_dir + 'sp500_vix.csv',
        columns=['S&P 500', 'VIX'],
        window_size=255,
        start_date='1992-01-02',
        end_date='2020-12-31',
        format='%Y/%m/%d',
        device=device,
```

To setup the model the parameters must be passed as well as the metrics being used to monitor the quality of training.

```python 
from ftsgan.ftsgan import FTSGAN
from ftsgan.evaluation import ACFLoss, HistoLoss, KurtosisLoss, LevEffLoss, SkewnessLoss, CorrelationLoss, SpreadLoss
params = {
    'g_in_channel': 6,
    'g_hidden_layers': [80] * 7,
    'g_skip_layer': 80,
    'd_hidden_layers': [80] * 7,
    'd_skip_layer': 80,
    'objective_features': ['acf_abs', 'acf_id', 'kurtosis'],
    'topk': 200
}
sp_data = dataset.series['S&P 500'].data.permute(0, 2, 1).contiguous()
sp500_metrics = [
    HistoLoss(sp_data, n_bins=200, name='abs_metric'),
    KurtosisLoss(sp_data, name='kurtosis'),
    SkewnessLoss(sp_data, name='skewness'),
    ACFLoss(sp_data, max_lag=64, name='acf_id', threshold=0.3),
    ACFLoss(sp_data, max_lag=64, name='acf_abs', transform=torch.abs, threshold=0.3),
    LevEffLoss(sp_data, name='lev_eff'),
]
vix_data = dataset.series['VIX'].data.permute(0, 2, 1).contiguous()
vix_metrics = [
    HistoLoss(vix_data, n_bins=200, name='abs_metric'),
    KurtosisLoss(vix_data, name='kurtosis'),
    SkewnessLoss(vix_data, name='skewness'),
    ACFLoss(vix_data, max_lag=64, name='acf_id', threshold=0.3),
    ACFLoss(vix_data, max_lag=64, name='acf_abs', transform=torch.abs, threshold=0.3),
    LevEffLoss(vix_data, name='lev_eff'),
]
all_metrics = [
    CorrelationLoss(sp_data, vix_data, name='corr_loss', threshold=0.04)
]
test_metrics = {0: sp500_metrics, 1: vix_metrics, 'all': all_metrics}
num_series = 2
batch_size=256
optimal_params = 'save'

model = FTSGAN(test_metrics, num_series=num_series, num_epochs=3000, batch_size=batch_size, params=params, save_dir=save_dir, ngpu=ngpu, verbose=5)
```

The ```optimal_params``` sets if the model will stop when a threshold is reached or will run for the set number of epochs and then save the best performing model.

The ```params``` overwrite the default implementation. The metrics must be chosen and passed to the ```test_metrics``` dictionary where each series is assigned a number which is used as the key.

To train the model it is as simple as calling the train method on the ```FTSGAN``` class:

```python
model.train(financial_series, optimal_params=optimal_params)
```

Once training is complete samples can be generated by the following:

```python
t_dim = dataset.series['S&P 500'].raw_data.shape[-1] - model.net_g.receptive_field_size()
synthetic_returns, synthetic_data, synthetic_paths  = model.generate_inverse(
    100,
    t_dim,
    dataset,
    optimal_params='save',
    pre_transform_dims={'S&P 500': (0, 2), 'VIX': (2)},
    accept_reject=False)
```
This will generate 100 samples of length given by ```t_dim```.
