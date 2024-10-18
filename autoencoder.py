import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import Counter


def train_autoencoder(model, train_loader, test_loader, device, log=print, num_epochs=300):
    calc_mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    def __init__(self, num_features, latent_features, reception_percent, padding, do_max_pool=False):
        super(PermutingConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool

        random_state = random.getstate()
        try:
            # random.seed(42)
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

        self.encoder = MultiEncoder(num_features, self.masks, padding, do_max_pool=do_max_pool)
        
        # Decoder
        if self.do_max_pool:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(latent_features, 32, kernel_size=3, padding=1 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.MaxUnpool1d(kernel_size=2),
                nn.ConvTranspose1d(32, 32, kernel_size=5, padding=2 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.MaxUnpool1d(kernel_size=2),
                nn.ConvTranspose1d(32, num_features, kernel_size=7, padding=3 if padding == 'same' else 0, output_padding=0)
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(latent_features, 32, kernel_size=3, padding=1 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 32, kernel_size=5, padding=2 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose1d(32, num_features, kernel_size=7, padding=3 if padding == 'same' else 0, output_padding=0)
            )
        
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.dropout(encoded, p=0.3, training=self.training)
        decoded = self.decoder(encoded)
        return decoded, encoded


class MultiEncoder(nn.Module):
    def __init__(self, num_features, masks, padding, do_max_pool=False):
        super(MultiEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.num_branches = len(masks)
        if do_max_pool:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=num_features * self.num_branches, out_channels=8 * self.num_branches, 
                        kernel_size=7, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(8 * self.num_branches),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool),
                nn.Conv1d(in_channels=8 * self.num_branches, out_channels=8 * self.num_branches, 
                        kernel_size=5, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(8 * self.num_branches),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool),
                nn.Conv1d(in_channels=8 * self.num_branches, out_channels=self.num_branches, 
                        kernel_size=3, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(self.num_branches),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=num_features * self.num_branches, out_channels=8 * self.num_branches, 
                        kernel_size=7, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(8 * self.num_branches),
                nn.ReLU(),
                nn.Conv1d(in_channels=8 * self.num_branches, out_channels=8 * self.num_branches, 
                        kernel_size=5, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(8 * self.num_branches),
                nn.ReLU(),
                nn.Conv1d(in_channels=8 * self.num_branches, out_channels=self.num_branches, 
                        kernel_size=3, padding=padding, groups=self.num_branches),
                nn.BatchNorm1d(self.num_branches),
                nn.ReLU()
            )

        self.masks = nn.Parameter(masks, requires_grad=False)
    
    def forward(self, x):
        # Apply masks
        x = x.repeat(1, self.num_branches, 1)
        x *= self.masks.view(1, -1).repeat(1, 1).unsqueeze(-1)

        # Apply encoder
        encoded = self.encoder(x)

        return encoded

    def set_return_indices(self, return_indices):
        if return_indices == self.return_indices:
            return

        self.return_indices = return_indices
        self.encoder[3] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class RegularConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, num_conv_filters=128):
        super(RegularConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.num_conv_filters = num_conv_filters
        
        self.encoder = RegularConvEncoder(num_features, latent_features, padding, do_max_pool=do_max_pool, num_conv_filters=num_conv_filters)
        if self.do_max_pool:
            self.decoder = nn.Sequential(
                nn.MaxUnpool1d(kernel_size=2),
                nn.ReLU(),
                nn.ConvTranspose1d(latent_features, self.num_conv_filters, kernel_size=self.encoder.kernel_sizes[2], padding=1 if padding == 'same' else 0, output_padding=0),
                nn.MaxUnpool1d(kernel_size=2),
                nn.ConvTranspose1d(self.num_conv_filters, self.num_conv_filters, kernel_size=self.encoder.kernel_sizes[1], padding=2 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.MaxUnpool1d(kernel_size=2),
                nn.ConvTranspose1d(self.num_conv_filters, num_features, kernel_size=self.encoder.kernel_sizes[0], padding=3 if padding == 'same' else 0, output_padding=0)
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(latent_features, self.num_conv_filters, kernel_size=self.encoder.kernel_sizes[2], padding=1 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose1d(self.num_conv_filters, self.num_conv_filters, kernel_size=self.encoder.kernel_sizes[1], padding=2 if padding == 'same' else 0, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose1d(self.num_conv_filters, num_features, kernel_size=self.encoder.kernel_sizes[0], padding=3 if padding == 'same' else 0, output_padding=0)
            )
        
    def forward(self, x):
        if self.return_indices:
            encoded, indices, sizes = self.encoder(x)
            encoded = F.dropout(encoded, p=0.3, training=self.training)
            indices = indices[::-1]
            sizes = sizes[::-1]
            decoded = encoded
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, nn.MaxUnpool1d):
                    decoded = layer(decoded, indices[i//3], output_size=sizes[i//3])
                else:
                    decoded = layer(decoded)
        else:
            encoded = self.encoder(x)
            encoded = F.dropout(encoded, p=0.3, training=self.training)
            decoded = self.decoder(encoded)
        return decoded, encoded


class RegularConvEncoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, num_conv_filters=32):
        super(RegularConvEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.num_conv_filters = num_conv_filters
        self.kernel_sizes = [7, 5, 3]

        if do_max_pool:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[0], padding=padding),
                nn.BatchNorm1d(num_conv_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool),
                nn.Conv1d(in_channels=num_conv_filters, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[1], padding=padding),
                nn.BatchNorm1d(num_conv_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool),
                nn.Conv1d(in_channels=num_conv_filters, out_channels=latent_features, kernel_size=self.kernel_sizes[2], padding=padding),
                nn.BatchNorm1d(latent_features),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[0], padding=padding),
                nn.BatchNorm1d(num_conv_filters),
                nn.ReLU(),
                nn.Conv1d(in_channels=num_conv_filters, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[1], padding=padding),
                nn.BatchNorm1d(num_conv_filters),
                nn.ReLU(),
                nn.Conv1d(in_channels=num_conv_filters, out_channels=latent_features, kernel_size=self.kernel_sizes[2], padding=padding),
                nn.BatchNorm1d(latent_features),
                nn.ReLU()
            )
        
    def forward(self, x):
        if self.return_indices:
            indices = []
            sizes = []
            for layer in self.encoder:
                if isinstance(layer, nn.MaxPool1d):
                    sizes.append(x.size())
                    x, idx = layer(x)
                    indices.append(idx)
                else:
                    x = layer(x)
            return x, indices, sizes
        else:
            return self.encoder(x)

    def set_return_indices(self, return_indices):
        if return_indices == self.return_indices:
            return

        self.return_indices = return_indices
        self.encoder[3] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
