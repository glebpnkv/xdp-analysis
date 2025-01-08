from torch import nn


class TimeConvolutionBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.2,
    ):

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=(kernel_size // 2)
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)
