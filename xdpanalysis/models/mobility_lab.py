from dataclasses import dataclass, field

import torch
from torch import nn

from ca_tcc.models.attention import SeqTransformer
from layers.conv_block import TimeConvolutionBlock


@dataclass
class MobilityLabConfigs:
    num_keys: int = 6  # Number of MobilityLab sensors (keys)
    input_channels: int = 15  # Number of MobilityLab features
    num_classes: int = 2  # Number of classes we are predicting

    num_key_channels: int = 128

    time_conv_blocks: list[dict] = field(default_factory=list)

    # Model
    features_len: int = 16
    kernel_size: int = 8
    stride: int = 1
    dropout: float = 0.2
    final_out_channels: int = 16
    d_model: int = 16
    num_heads: int = 4
    dff: int = 64
    num_transformer_layers: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100

    beta1 = 0.9
    beta2 = 0.99

    tc_prediction_timesteps: int = 5
    temperature: float = 0.15
    use_cosine_similarity: bool = True

    def __post_init__(self):
        if not self.time_conv_blocks:
            self.time_conv_blocks = [
                {"input_channels": self.input_channels, "out_channels": 64},
                {"input_channels": 64, "out_channels": 32},
                {"input_channels": 32, "out_channels": 16},
            ]
        self.num_time_conv_blocks = len(self.time_conv_blocks)


class MobilityLabSeqTransformer(SeqTransformer):
    def __init__(
        self,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=1,
        dropout=0.1
    ):
        super(MobilityLabSeqTransformer, self).__init__(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dropout=dropout
        )

    def forward(self, x):
        x = self.patch_to_embedding(x)
        c_tokens = torch.repeat_interleave(
            input=self.c_token,
            repeats=x.shape[0],
            dim=0
        )
        x = torch.cat((c_tokens, x), dim=1)

        # Adding positional encoding
        x = self.pos_encoding(x)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])

        return x, c_t


class MobilityLabModel(nn.Module):
    def __init__(
        self,
        configs: MobilityLabConfigs
    ):
        super(MobilityLabModel, self).__init__()

        self.tc_prediction_timesteps = configs.tc_prediction_timesteps

        # Layer for processing multiple sensors (keys) from MobilityLab
        self.conv_keys = nn.Sequential(
            nn.Conv2d(
                in_channels=configs.num_keys,
                out_channels=configs.num_key_channels,
                kernel_size=(8, configs.input_channels),
                padding='same'
            ),
            nn.BatchNorm2d(configs.num_key_channels),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        self.time_conv_blocks = nn.ModuleList(
            [
                TimeConvolutionBlock(
                    input_channels=cur_block["input_channels"],
                    out_channels=cur_block["out_channels"],
                    kernel_size=configs.kernel_size,
                    stride=configs.stride,
                    dropout=configs.dropout
                )
                for cur_block in configs.time_conv_blocks
            ]
        )
        self.seq_transformer = MobilityLabSeqTransformer(
            patch_size=configs.final_out_channels,
            dim=configs.d_model,
            depth=configs.num_transformer_layers,
            heads=configs.num_heads,
            mlp_dim=configs.dff,
            dropout=configs.dropout
        )

        self.projection_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.BatchNorm1d(configs.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.d_model // 2, configs.d_model // 4),
        )

        self.wk = nn.ModuleList([
            nn.Linear(configs.d_model, configs.d_model)
            for _ in range(configs.tc_prediction_timesteps)
        ])

        self.logits = nn.Linear(configs.d_model // 4, configs.num_classes)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Averaging the signal over multiple sensors
        x = self.conv_keys(x)
        x = torch.mean(x, dim=1)

        # Applying time convolutions
        x = torch.swapaxes(x, 1, 2)
        for time_conv_block in self.time_conv_blocks:
            x = time_conv_block(x)

        x = torch.swapaxes(x, 1, 2)

        x, c_t = self.seq_transformer(x)

        c_t_proj = self.projection_head(c_t)
        logits = self.logits(c_t_proj)

        return c_t, logits, x

    def forward_tc(
        self,
        z_aug1: torch.Tensor,
        z_aug2: torch.Tensor
    ) -> torch.Tensor:
        batch = z_aug1.shape[0]
        seq_len = z_aug1.shape[1]

        t_samples = torch.randint(
            seq_len - self.tc_prediction_timesteps, size=(1,)
        ).long().to(self.device)

        encode_samples = z_aug2[:, t_samples:t_samples + self.tc_prediction_timesteps, :].to(self.device)
        forward_seq = z_aug1[:, :t_samples, :].to(self.device)

        # Getting c_t token from forward_seq
        _, c_t = self.seq_transformer(forward_seq)

        pred = torch.stack(
            [
                layer(c_t)
                for layer in self.wk
            ],
            dim=1
        )

        nce = torch.einsum("iak,jak->aij", encode_samples, pred)
        nce = torch.diagonal(self.lsoftmax(nce), 0, 1, 2)
        nce = -1.0 * torch.sum(nce) / (batch * self.tc_prediction_timesteps)

        return nce, self.projection_head(c_t)
