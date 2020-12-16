from torch import nn

threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

class PatchModel(nn.Module):
    """
    Model that tells if a 16 x 16 RGB (as a 3 x 16 x 16 tensor) corresponds to a road (1) or not (0)
    """
    def __init__(self):
        super().__init__()
        
        # 3 channels 16 x 16
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5
        )

        # 32 channels 12 x 12 (12 = 16 - (kernel_size - 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 32 channels 6 x 6 (6 = 12 / kernel_size)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        # 64 channels 4 x 4 (4 = 6 - (kernel_size - 1))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 64 channels 2 x 2 (2 = 4 / kernel_size)
        self.lin1 = nn.Linear(
            in_features=64*2*2,
            out_features= 512
        )
        self.lin2 = nn.Linear(
            in_features=512,
            out_features=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64*2*2)
        x = self.relu(self.lin1(x))
        x = self.sigmoid(self.lin2(x))
        # If we are in testing mode then the output should be either 0 or 1
        if not self.training:
            x = 1 * (x > threshold)
        return x.view(-1)