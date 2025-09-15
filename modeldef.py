import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Nbeats model definition
class NbeatsBlock (nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers) -> None:
        '''
            input_size: int, size of the input
            hidden_size: int, size of the hidden layers
            output_size: int, size of the output
            n_layers: int, number of layers
        '''
        super().__init__()

        self.fc = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size if i < n_layers else output_size),
                nn.ReLU()
            ) for i in range(n_layers + 1)]
        )

    def forward(self, x):
        return self.fc(x)

class Nbeats(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=4, n_stacks=3) -> None:
        '''
            input_size: int, size of the input
            hidden_size: int, size of the hidden layers
            output_size: int, size of the output
            n_layers: int, number of layers
            n_stacks: int, number of stacks
        '''
        super().__init__()

        self.stacks = nn.ModuleList(
            [NbeatsBlock(input_size, hidden_size, output_size, n_layers) for _ in range(n_stacks)]
        )

    def forward(self, x):
        forcast = sum(stack(x) for stack in self.stacks)
        return forcast

# timesnet model definition
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # THE FIX IS HERE: We need to provide x_mark to use temporal_embedding
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.cat(res_list, dim=1)
        return res

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff * configs.num_kernels, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetModel(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(TimesNetModel, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        # ... (rest of the Model class)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(stdev)
        dec_out = dec_out.add(means)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # ... (rest of the forward method)
        return None

def create_time_features(df, time_col='Date', freq='d'):
    """
    Creates time series features from a datetime index.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['weekday'] = df[time_col].dt.dayofweek
    # Add more features if needed, like 'weekofyear'

    # The TimeFeatureEmbedding expects these features in a specific order based on `freq`
    if freq == 'd': # Daily
        feature_cols = ['month', 'day', 'weekday']
    else: # Add more logic for other frequencies if needed
        raise NotImplementedError(f"Frequency '{freq}' not implemented.")

    return df[feature_cols].values

# This function creates input/output sequences
def create_sequences(input_data, time_features, seq_len, pred_len):
    X, y, X_mark = [], [], []
    for i in range(len(input_data) - seq_len - pred_len + 1):
        X.append(input_data[i:(i + seq_len)])
        y.append(input_data[i + seq_len:i + seq_len + pred_len])
        X_mark.append(time_features[i:(i + seq_len)])
    return np.array(X), np.array(y), np.array(X_mark)

# residual ensemble model definition
class ResidualEnsemble(nn.Module):
    def __init__(self, nbeats_configs, nbeats_model_path, timesnet_configs, timesnet_model_path):
        super(ResidualEnsemble, self).__init__()
        self.nbeats_model = Nbeats(**nbeats_configs)
        self.timesnet_model = TimesNetModel(timesnet_configs)

        self.nbeats_model.load_state_dict(torch.load(nbeats_model_path))
        self.timesnet_model.load_state_dict(torch.load(timesnet_model_path))

        self.nbeats_model.eval()
        self.timesnet_model.eval()

    def forward(self, x_enc, x_mark_enc):
        with torch.inference_mode():
            # nbeats prediction
            nbeats_input = x_enc.view(x_enc.size(0), -1) # [B, seq_len]
            nbeats_out_flat = self.nbeats_model(nbeats_input) # [B, pred_len]
            nbeats_out = nbeats_out_flat.unsqueeze(-1) # [B, pred_len, 1]
            
            # timesnet prediction on residuals
            dec_input = torch.zeros_like(nbeats_out).to(x_enc.device)
            residual_out = self.timesnet_model(x_enc, x_mark_enc, dec_input, dec_input) # [B, pred_len, 1]

            # Final prediction
            final_out = nbeats_out + residual_out
            return final_out


class RegimeDetector:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df):
        df['logreturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volume'] = df['logreturns'].rolling(window=14).std().fillna(method='bfill')
        X = df[['logreturns', 'volume', 'residuals_train']].fillna(0).values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True

    def detect(self, new_obs):
        if not self.fitted:
            raise ValueError("RegimeDetector must be fitted first.")
        
        new_obs_scaled = self.scaler.transform(new_obs.reshape(1,-1))
        # Posterior over states
        logprob, posteriors = self.model.score_samples(new_obs_scaled)
        current_state = np.argmax(posteriors)
        return current_state, posteriors