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
        """
        Initialises a TimeConvolutionBlock.

        The block includes a sequence of layers: convolution, batch normalization,
        ReLU activation, max pooling, and dropout. The convolution layer applies
        a 1D convolution operation with specified input and output channels, kernel
        size, stride, and zero padding to preserve dimensions. Batch normalization
        is applied to stabilize training, followed by ReLU activation to introduce
        non-linearity. A max pooling layer reduces the spatial dimensions of the
        feature maps. Finally, dropout is applied to prevent overfitting.

        Arguments:
        ----------
        input_channels : int
            Number of input channels to the convolutional layer.

        out_channels : int
            Number of output channels after the convolutional layer.

        kernel_size : int
            The size of the convolutional kernel.

        stride : int, optional
            The stride of the convolutional operation. Default is 1.

        dropout : float, optional
            The dropout rate to apply after the max pooling layer. Default is 0.2.
        """
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
        """
        Passes the input tensor through the predefined block.

        The method applies the functionality of the `block` attribute to the input
        tensor `x`. It serves as the forward method, executing the core processing
        logic defined within the class.

        Args:
            x: Input tensor passed to the block for processing.

        Returns:
            The output tensor after being processed by the block.
        """
        return self.block(x)
