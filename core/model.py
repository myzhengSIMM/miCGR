import torch
import torch.nn as nn
import torch.nn.functional as F
# import tqdm


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes= (1, 3, 5), stride=1):
        super(Inception, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.paddings = [(ks - 1) // 2 for ks in self.kernel_sizes]
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_sizes[0], stride=self.stride, padding=self.paddings[0])
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_sizes[1], stride=self.stride, padding=self.paddings[1])
        self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_sizes[2], stride=self.stride, padding=self.paddings[2])
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x1 = self.bn1(F.relu(self.conv1(x)))
        x2 = self.bn2(F.relu(self.conv2(x)))
        x3 = self.bn3(F.relu(self.conv3(x)))
        x = torch.cat([x1, x2, x3], dim=1)
        return x



class StackedConv_k2(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k2, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 4*4
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 4*4
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 1, 3), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 4*4
        self.dropout= nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 2*2*60
        return x

class StackedConv_k3(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k3, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 8*8
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 4*4
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 1, 3), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 2*2*60
        return x

class StackedConv_k4(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k4, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 16*16
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 8*8
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 3, 5), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 4*4*60
        return x


class StackedConv_k5(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k5, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 32*32
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=4, padding=2) # 8*8
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 3, 5), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 4*4*60
        return x

class StackedConv_k6(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k6, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 64*64
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=4, padding=2) # 16*16
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 5, 9), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=4, padding=2) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 4*4*60
        return x

class StackedConv_k7(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k7, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 128*128
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=4)
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 5, 9), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=4, padding=2) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 4*4*60
        return x

class StackedConv_k8(nn.Module):
    """
    learn from MolMapNet
    """
    def __init__(self, in_channels=1, kernel_size=3, dropout=0.5):
        super(StackedConv_k8, self).__init__()
        # input: 1 * 16 * 16
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=30,
                               kernel_size=kernel_size, padding=padding) # 256*256
        # self.conv1 = nn.DataParallel(self.conv1, device_ids=[0,1])
        self.bn1 = nn.BatchNorm2d(30)
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=4) # 32*32
        self.inception1 = Inception(in_channels=30, out_channels=20, kernel_sizes=(1, 9, 17), stride=1)
        # self.inception1 = nn.DataParallel(self.inception1, device_ids=[0,1])
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=8, padding=4) # 4*4
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.inception1(x)
        x = self.dropout(self.pool2(x))
        x = torch.flatten(x, start_dim=1) # 4*4*60
        return x

class NSWSSC_k2(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k2, self).__init__()
        self.sac1 = StackedConv_k2(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k2(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)

class NSWSSC_k3(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k3, self).__init__()
        self.sac1 = StackedConv_k3(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k3(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)


class NSWSSC_k4(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k4, self).__init__()
        self.sac1 = StackedConv_k4(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k4(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)

class NSWSSC_k5(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k5, self).__init__()
        self.sac1 = StackedConv_k5(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k5(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)


class NSWSSC_k6(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k6, self).__init__()
        self.sac1 = StackedConv_k6(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k6(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)

class NSWSSC_k7(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k7, self).__init__()
        self.sac1 = StackedConv_k7(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k7(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)

class NSWSSC_k8(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k8, self).__init__()
        self.sac1 = StackedConv_k8(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k8(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4*4*60*2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, input_lst):
        x1 = self.sac1(input_lst[0])
        x2 = self.sac2(input_lst[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x

    def inference(self, input_lst, batch_size, device):
        sample_num = input_lst[0].shape[0]
        y = torch.zeros(sample_num, 2)
        for start in range(0, sample_num, batch_size):
            end = start + batch_size
            x1 = input_lst[0][start:end].to(device)
            x2 = input_lst[1][start:end].to(device)
            x1 = self.sac1(x1)
            x2 = self.sac2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn1(F.relu(self.dense1(x)))
            x = self.bn2(F.relu(self.dense2(x)))
            x = self.bn3(F.relu(self.dense3(x)))
            x = self.dropout(self.bn4(F.relu(self.dense4(x))))
            x = self.dense5(x)
            y[start:end] = x.cpu()
        return y.to(device)

class NSWSSC_k6_plot(nn.Module):
    def __init__(self, dropout=0.5):
        super(NSWSSC_k6_plot, self).__init__()
        self.sac1 = StackedConv_k6(in_channels=3, dropout=dropout)
        self.sac2 = StackedConv_k6(in_channels=1, dropout=dropout)
        self.dense1 = nn.Linear(4 * 4 * 60 * 2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dense2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dense3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dense5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dense1.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.kaiming_uniform_(self.dense2.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.kaiming_uniform_(self.dense3.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.kaiming_uniform_(self.dense4.weight.data, nonlinearity='relu')
        nn.init.constant_(self.dense4.bias, 0)
        nn.init.uniform_(self.dense5.weight.data, a=-1.0, b=1.0)
        nn.init.constant_(self.dense5.bias, 0)

    def forward(self, inputs):
        x1 = inputs[:, :3, :, :]
        x2 = inputs[:, 3, :, :].unsqueeze(1)
        x1 = self.sac1(x1)
        x2 = self.sac2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn1(F.relu(self.dense1(x)))
        x = self.bn2(F.relu(self.dense2(x)))
        x = self.bn3(F.relu(self.dense3(x)))
        x = self.dropout(self.bn4(F.relu(self.dense4(x))))
        x = self.dense5(x)
        return x



