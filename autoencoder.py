import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import Counter


def train_autoencoder(model, train_loader, test_loader, device, log=print):
    calc_mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 300
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model.to(device)
    
    for epoch in range(1, num_epochs+1):
        # training
        model.train()

        to_print = f'epoch: {epoch:4d}/{num_epochs} '

        n_examples = 0
        train_loss = 0
        for (data, _) in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(data)

            loss = calc_mse_loss(outputs, data)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
        
            n_examples += data.size(0)

        train_loss /= n_examples
        scheduler.step()
    
        # testing
        if epoch % 10 == 0:
            test_loss = 0
            n_examples = 0
            model.eval()
            for (data, _) in test_loader:
                data = data.to(device)
                
                with torch.no_grad():
                    outputs, _ = model(data)

                    loss = calc_mse_loss(outputs, data)
        
                n_examples += data.size(0)
                test_loss += loss.item() * data.size(0)
            
            test_loss /= n_examples

            to_print += f'mse loss: {test_loss:>5.4f}'
            log(to_print, flush=True)



class PermutingConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, reception_percent, padding):
        super(PermutingConvAutoencoder, self).__init__()

        random_state = random.getstate()
        try:
            random.seed(42)
            for _ in range(100):
                # It may happen that feature is not taken into account at all, or it's
                # taken into account by all the latent features, let's regenerate
                # permutations in such cases. 100 attempts, after that we raise an error.
                self.receive_from = []
                input_features_per_latent = max(int(reception_percent * num_features), 1)
                for _ in range(latent_features):
                    curr_receive_from = random.sample(range(num_features), input_features_per_latent)
                    curr_receive_from.sort()
                    self.receive_from.append(curr_receive_from)
                counter = Counter([item for curr_receive_from in self.receive_from for item in curr_receive_from])
                if len(counter) == num_features and all(cnt != latent_features for cnt in counter.values()):
                    break
            else:
                raise RuntimeError('Could not generate satisfying permutations, aborting')
        finally:
            random.setstate(random_state)
        
        self.masks = nn.Parameter(torch.FloatTensor([[1 if i in curr_receive_from else 0 for i in range(num_features)] for curr_receive_from in self.receive_from]), requires_grad=False)

        self.encoder = MultiEncoder(num_features, self.masks, padding)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_features, 256, kernel_size=3, padding=1 if padding == 'same' else 0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=5, padding=2 if padding == 'same' else 0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256, num_features, kernel_size=7, padding=3 if padding == 'same' else 0, output_padding=0)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.dropout(encoded, p=0.3, training=self.training)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class MultiEncoder(nn.Module):
    def __init__(self, num_features, masks, padding):
        super(MultiEncoder, self).__init__()
        
        self.num_branches = len(masks)
        self.encoder = GroupedBranchEncoder(num_features, self.num_branches, padding)

        self.masks = nn.Parameter(masks, requires_grad=False)
    
    def forward(self, x):
        # Apply masks
        x = x.repeat(1, self.num_branches, 1)
        x *= self.masks.view(1, -1).repeat(1, 1).unsqueeze(-1)

        # Apply encoder
        encoded = self.encoder(x)

        return encoded

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class GroupedBranchEncoder(nn.Module):
    def __init__(self, num_features, num_branches, padding):
        super(GroupedBranchEncoder, self).__init__()
        
        self.num_branches = num_branches

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features * num_branches, out_channels=32 * num_branches, 
                      kernel_size=7, padding=padding, groups=num_branches),
            nn.BatchNorm1d(32 * num_branches),
            nn.ReLU(),
            nn.Conv1d(in_channels=32 * num_branches, out_channels=32 * num_branches, 
                      kernel_size=5, padding=padding, groups=num_branches),
            nn.BatchNorm1d(32 * num_branches),
            nn.ReLU(),
            nn.Conv1d(in_channels=32 * num_branches, out_channels=num_branches, 
                      kernel_size=3, padding=padding, groups=num_branches),
            nn.BatchNorm1d(num_branches),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class RegularConvEncoder(nn.Module):
    def __init__(self, num_features, latent_features, padding):
        super(RegularConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=7, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=latent_features, kernel_size=3, padding=padding),
            nn.BatchNorm1d(latent_features),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
