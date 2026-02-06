import torch
import torch.nn as nn
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(4, 48, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(48, 72, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(72)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(72, 108, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(108)
        self.relu3 = nn.ReLU(inplace=True)

        self.capsule1 = CapsuleLayer(1, 108, 64, 64)
        self.capsule2 = CapsuleLayer(1, 1, 64, 64)
        self.capsule3 = CapsuleLayer(1, 1, 64, 64)

        self.deconv1 = nn.ConvTranspose2d(108, 72, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(72)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(72, 48, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.relu5 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, label):
        x = torch.cat([x, torch.full_like(x[:, :1, :, :], label, dtype=torch.float32)], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.capsule1(x)
        x = self.capsule2(x)
        x = self.capsule3(x)

        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.deconv3(x)
        x = self.tanh(x)

        return x

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_capsule_dim, out_capsule_dim, num_iterations=1):
        super(CapsuleLayer, self).__init__()
        self.num_iterations = num_iterations
        self.weights = nn.Parameter(torch.randn(out_capsules, in_capsules, in_capsule_dim, out_capsule_dim))

    def forward(self, x):
        batch_size = x.size(0)
        u_hat = torch.matmul(self.weights, x)
        b = torch.zeros(batch_size, 1, x.size(3), 1).to(x.device)

        for _ in range(self.num_iterations):
            c = torch.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self.squash(s)
            b = b + (u_hat * v).sum(dim=-1, keepdim=True)
            
        return v.permute(1,0,2,3)+x

    def squash(self, v):
        mag_sq = torch.sum(v**2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq)
        v = (mag_sq / (1.0 + mag_sq)) * (v / mag)
        return v
    

device = "cuda" if torch.cuda.is_available() else "cpu"    
print(device)

generator = Generator()
#generator = torch.nn.DataParallel(generator)
generator = generator.to(device)
checkpoint = torch.load(r'weights/capsule.ckpt', map_location=device)
generator.load_state_dict(checkpoint['generator'])
