import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultiEncoder(nn.Module):
    def __init__(self, num_features, masks, padding, do_max_pool=False, do_batch_norm=True):
        super(MultiEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.num_branches = len(masks)

        layers = []
        layers.append(
            nn.Conv1d(
                in_channels=num_features * self.num_branches,
                out_channels=16 * self.num_branches,
                kernel_size=7,
                padding=padding,
                groups=self.num_branches,
            )
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(16 * self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(
            nn.Conv1d(
                in_channels=16 * self.num_branches,
                out_channels=16 * self.num_branches,
                kernel_size=5,
                padding=padding,
                groups=self.num_branches,
            )
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(16 * self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(
            nn.Conv1d(
                in_channels=16 * self.num_branches,
                out_channels=self.num_branches,
                kernel_size=3,
                padding=padding,
                groups=self.num_branches,
            )
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))

        self.encoder = nn.Sequential(*layers)

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
        self.encoder[3 if self.do_batch_norm else 2] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7 if self.do_batch_norm else 5] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11 if self.do_batch_norm else 8] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class PermutingConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, reception_percent, padding, do_max_pool=False, do_batch_norm=True):
        super(PermutingConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.do_batch_norm = do_batch_norm

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
                raise RuntimeError("Could not generate satisfying permutations, aborting")
        finally:
            random.setstate(random_state)

        self.masks = nn.Parameter(
            torch.FloatTensor([[1 if i in curr_receive_from else 0 for i in range(num_features)] for curr_receive_from in self.receive_from]),
            requires_grad=False,
        )

        self.encoder = MultiEncoder(num_features, self.masks, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm)

        # Decoder
        layers = []
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(latent_features, 32, kernel_size=3, padding=1 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(32, 32, kernel_size=5, padding=2 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(32, num_features, kernel_size=7, padding=3 if padding == "same" else 0, output_padding=0))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.dropout(encoded, p=0.3, training=self.training)
        decoded = self.decoder(encoded)
        return decoded, encoded


class RegularConvEncoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=32):
        super(RegularConvEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.do_batch_norm = do_batch_norm
        self.num_conv_filters = num_conv_filters
        self.kernel_sizes = [7, 5, 3]

        layers = []

        layers.append(nn.Conv1d(in_channels=num_features, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[0], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(num_conv_filters))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(nn.Conv1d(in_channels=num_conv_filters, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[1], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(num_conv_filters))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(nn.Conv1d(in_channels=num_conv_filters, out_channels=latent_features, kernel_size=self.kernel_sizes[2], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(latent_features))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))

        self.encoder = nn.Sequential(*layers)

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
        self.encoder[3 if self.do_batch_norm else 2] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7 if self.do_batch_norm else 5] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11 if self.do_batch_norm else 8] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


def build_decoder(num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True):
    layers = []
    if do_max_pool:
        layers.append(nn.MaxUnpool1d(kernel_size=2))
    layers.append(nn.ConvTranspose1d(latent_features, 32, kernel_size=3, padding=1 if padding == "same" else 0, output_padding=0))
    layers.append(nn.ReLU())
    if do_max_pool:
        layers.append(nn.MaxUnpool1d(kernel_size=2))
    layers.append(nn.ConvTranspose1d(32, 32, kernel_size=5, padding=2 if padding == "same" else 0, output_padding=0))
    layers.append(nn.ReLU())
    if do_max_pool:
        layers.append(nn.MaxUnpool1d(kernel_size=2))
    layers.append(nn.ConvTranspose1d(32, num_features, kernel_size=7, padding=3 if padding == "same" else 0, output_padding=0))

    return nn.Sequential(*layers)


class RegularConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=32):
        super(RegularConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.return_indices = do_max_pool
        self.num_conv_filters = num_conv_filters

        self.encoder = RegularConvEncoder(
            num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm, num_conv_filters=num_conv_filters
        )

        self.decoder = build_decoder(num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm)

    def forward(self, x):
        if self.return_indices:
            encoded, indices, sizes = self.encoder(x)
            encoded = F.dropout(encoded, p=0.3, training=self.training)
            indices = indices[::-1]
            sizes = sizes[::-1]
            decoded = encoded
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, nn.MaxUnpool1d):
                    decoded = layer(decoded, indices[i // 3], output_size=sizes[i // 3])
                else:
                    decoded = layer(decoded)
        else:
            encoded = self.encoder(x)
            encoded = F.dropout(encoded, p=0.3, training=self.training)
            decoded = self.decoder(encoded)
        return decoded, encoded


def train_autoencoder(model, train_loader, test_loader, device, log=print, num_epochs=300):
    calc_mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()

        to_print = f"epoch: {epoch:4d}/{num_epochs} "

        n_examples = 0
        train_loss = 0
        for data, _ in train_loader:
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

        # Validation
        if epoch % 10 == 0:
            test_loss = 0
            n_examples = 0
            model.eval()
            for data, _ in test_loader:
                data = data.to(device)

                with torch.no_grad():
                    outputs, _ = model(data)

                    loss = calc_mse_loss(outputs, data)

                n_examples += data.size(0)
                test_loss += loss.item() * data.size(0)

            test_loss /= n_examples

            to_print += f"mse loss: {test_loss:>5.4f}"
            log(to_print, flush=True)


class VariationalEncoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=32):
        super(VariationalEncoder, self).__init__()

        # Shared convolutional encoder
        self.encoder_base = RegularConvEncoder(
            num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm, num_conv_filters=num_conv_filters
        )
        
        # Two heads: mean (mu) and log-variance (logvar)
        self.mu_head = nn.Conv1d(latent_features, latent_features, kernel_size=1)
        self.logvar_head = nn.Conv1d(latent_features, latent_features, kernel_size=1)
        
        self.encoder = nn.Sequential(
            self.encoder_base,
            self.mu_head,
        )

    def forward(self, x):
        return self.encoder(x)
    
    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class VariationalConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=32):
        super(VariationalConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.return_indices = do_max_pool
        self.num_conv_filters = num_conv_filters

        self.encoder = VariationalEncoder(
            num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm, num_conv_filters=num_conv_filters
        )

        # Decoder (same as in RegularConvAutoencoder)
        self.decoder = build_decoder(num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # Encode
        if self.return_indices:
            enc, indices, sizes = self.encoder.encoder_base(x)
        else:
            enc = self.encoder.encoder_base(x)

        # Compute mu and logvar
        mu = self.encoder.mu_head(enc)
        logvar = self.encoder.logvar_head(enc)

        if self.return_indices:
            return mu, logvar, indices, sizes
        return mu, logvar

    def forward(self, x):
        if self.return_indices:
            mu, logvar, indices, sizes = self.encode(x)
        else:
            mu, logvar = self.encode(x)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode
        if self.return_indices:
            # unpool indices are reversed order from encoder
            indices = indices[::-1]
            sizes = sizes[::-1]
            decoded = z
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, nn.MaxUnpool1d):
                    decoded = layer(decoded, indices[i // 3], output_size=sizes[i // 3])
                else:
                    decoded = layer(decoded)
        else:
            decoded = self.decoder(z)

        return decoded, mu, logvar


def train_vae_with_anchors(
    model,
    train_loader,      # yields (data, class_idx)
    test_loader,       # yields (data, class_idx)
    expert_protos,     # dict: expert_protos[class_idx][proto_idx] = numpy array (C, T)
    device,
    log=print,
    num_epochs=300,
    beta=1.0,          # KL weight
    gamma=0.1,        # anchor loss weight
    anchor_warmup=20   # epochs before we start using anchors
):
    recon_loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model.to(device)

    if expert_protos:
        st_key = list(k for k in expert_protos.keys() if expert_protos[k])[0]
        st_proto_key = list(expert_protos[st_key].keys())[0]
        proto_window_size = expert_protos[st_key][st_proto_key].shape[1]

    def compute_anchors():
        model.eval()
        anchors = {}
        with torch.no_grad():
            for cls, proto_dict in expert_protos.items():
                all_patches = []
                for proto in proto_dict.values():
                    x = torch.tensor(proto, dtype=torch.float32, device=device).unsqueeze(0)
                    _, mu, _ = model(x)
                    # mu: (1, C', T')
                    patches = mu.unfold(2, proto_window_size, 1)  # (1, C', num_patches, W)
                    b, c, npatches, w = patches.shape
                    patches = patches.permute(0,2,1,3).reshape(npatches, c*w)
                    all_patches.append(patches)
                if all_patches:
                    anchors[cls] = torch.cat(all_patches, dim=0)
        return anchors

    for epoch in range(1, num_epochs + 1):
        anchors = None
        if epoch > anchor_warmup and expert_protos:
            anchors = compute_anchors()

        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs, mu, logvar = model(x)

            # VAE loss
            recon = recon_loss_fn(outputs, x)
            kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl

            # anchor loss
            if anchors is not None:
                patches = mu.unfold(2, proto_window_size, 1)
                b, c, npatches, w = patches.shape
                patches = patches.permute(0,2,1,3).reshape(b, npatches, c*w)

                anchor_loss = 0.0
                for i in range(b):
                    cls = int(y[i].item())
                    anchor_patches = anchors.get(cls)
                    if anchor_patches is None:
                        continue
                    diffs = patches[i].unsqueeze(1) - anchor_patches.unsqueeze(0)
                    d2 = torch.sum(diffs * diffs, dim=2)
                    anchor_loss += torch.min(d2)
                loss = loss + gamma * anchor_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        # logging
        if epoch % 10 == 0:
            model.eval()
            test_loss = 0.0
            test_kl_loss = 0.0
            test_recon_loss = 0.0
            test_anchor_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs, mu, logvar = model(x)

                    recon = recon_loss_fn(outputs, x)
                    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss  = recon + beta * kl

                    if expert_protos:
                        anchors = compute_anchors()

                        patches = mu.unfold(2, proto_window_size, 1)
                        b, c, npatches, w = patches.shape
                        patches = patches.permute(0,2,1,3).reshape(b, npatches, c*w)

                        anchor_loss = 0.0
                        for i in range(b):
                            cls = int(y[i].item())
                            anchor_patches = anchors.get(cls)
                            if anchor_patches is None:
                                continue
                            diffs = patches[i].unsqueeze(1) - anchor_patches.unsqueeze(0)
                            d2 = torch.sum(diffs * diffs, dim=2)
                            anchor_loss += torch.min(d2)

                        if epoch > anchor_warmup:
                            loss = loss + gamma * anchor_loss
                        test_anchor_loss += anchor_loss.item()

                    test_loss += loss.item()
                    test_kl_loss += kl.item()
                    test_recon_loss += recon.item()

            log(f"Epoch {epoch}/{num_epochs}", flush=True)
            log(f"  Test Recon Loss: {test_recon_loss / len(test_loader.dataset):.4f}", flush=True)
            log(f"  Test KL Loss: {test_kl_loss / len(test_loader.dataset):.4f}", flush=True)
            if expert_protos:
                if epoch > anchor_warmup:
                    log(f"  Test Anchor Loss: {test_anchor_loss / len(test_loader.dataset):.4f}", flush=True)
                else:
                    log(f"  Test Anchor Loss: {test_anchor_loss / len(test_loader.dataset):.4f} (unregularized)", flush=True)
            log(f"  Test Total Loss: {test_loss / len(test_loader.dataset):.4f}", flush=True)
            log(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}", flush=True)


def train_vae_with_contrastive(
    model,
    train_loader,      # yields (data, class_idx)
    test_loader,       # yields (data, class_idx)
    expert_protos,     # dict: expert_protos[class_idx][proto_idx] = numpy array (C, T)
    device,
    log=print,
    num_epochs=300,
    beta=1.0,          # KL weight
    gamma=0.1,         # contrastive weight
    anchor_warmup=20,  # epochs before contrastive starts
    tau=0.1            # temperature for InfoNCE
):
    recon_loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)
    model.to(device)

    if expert_protos:
        st_key = list(k for k in expert_protos.keys() if expert_protos[k])[0]
        st_proto_key = list(expert_protos[st_key].keys())[0]
        proto_window_size = expert_protos[st_key][st_proto_key].shape[1]

    def compute_anchors_means():
        model.eval()
        means = {}
        with torch.no_grad():
            for cls, proto_dict in expert_protos.items():
                patches = []
                for proto in proto_dict.values():
                    x = torch.tensor(proto, dtype=torch.float32, device=device).unsqueeze(0)
                    _, mu, _ = model(x)  # mu: (1, C', T')
                    p = mu.unfold(2, proto_window_size, 1)  # (1,C',npatches,W)
                    b,c,npatches,w = p.shape
                    p = p.permute(0,2,1,3).reshape(npatches, c*w)
                    patches.append(p)
                if patches:
                    all_p = torch.cat(patches, dim=0)
                    means[cls] = all_p.mean(dim=0)  # (D,)
        return means

    anchors = None
    class_list = None
    for epoch in range(1, num_epochs + 1):
        if epoch > anchor_warmup:
            anchors = compute_anchors_means()
            class_list = sorted(anchors.keys())
            anchor_means_all = torch.stack([anchors[k] for k in class_list], dim=0).to(device)
            cls_to_idx = {k: i for i, k in enumerate(class_list)}

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs, mu, logvar = model(x)
            # Reconstruction + KL
            recon = recon_loss_fn(outputs, x)
            kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl

            # Contrastive patch loss
            if anchors is not None:
                p = mu.unfold(2, proto_window_size, 1)  # (B,C',npatches,W)
                b,c,npatches,w = p.shape
                z = p.permute(0,2,1,3).reshape(b, npatches, c*w)
                # mean over patches to get one embedding per sample
                z_mean = z.mean(dim=1)  # (B, D)
                # compute logits vs all class anchors
                logits = (z_mean @ anchor_means_all.t()) / tau  # (B, K)
                # labels map
                labels = torch.tensor([cls_to_idx.get(int(y_i), -1) for y_i in y], device=device)
                mask = labels >= 0
                if mask.any():
                    ce = F.cross_entropy(logits[mask], labels[mask])
                    loss = loss + gamma * ce

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Logging
        if epoch % 10 == 0:
            model.eval()
            t_recon, t_kl, t_contra, t_count = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs, mu, logvar = model(x)
                    recon = recon_loss_fn(outputs, x)
                    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    total = recon + beta * kl

                    t_recon += recon.item()
                    t_kl    += kl.item()
                    if anchors is not None:
                        p = mu.unfold(2, proto_window_size, 1)
                        b,c,npatches,w = p.shape
                        z = p.permute(0,2,1,3).reshape(b, npatches, c*w)
                        z_mean = z.mean(dim=1)
                        logits = (z_mean @ anchor_means_all.t()) / tau
                        labels = torch.tensor([cls_to_idx.get(int(y_i), -1) for y_i in y], device=device)
                        mask = labels >= 0
                        if mask.any():
                            ce = F.cross_entropy(logits[mask], labels[mask])
                            total = total + gamma * ce
                            t_contra += ce.item() * mask.sum().item()
                            t_count += mask.sum().item()
                    t_count = max(t_count, 1)

            log(f"Epoch {epoch}/{num_epochs}", flush=True)
            log(f"  Recon: {t_recon/len(test_loader.dataset):.4f}", flush=True)
            log(f"  KL:    {t_kl/len(test_loader.dataset):.4f}", flush=True)
            if anchors is not None and t_count>0:
                log(f"  Contrastive: {t_contra/t_count:.4f}", flush=True)
            log(f"  Total: { (t_recon+t_kl+(t_contra if t_count>0 else 0))/len(test_loader.dataset):.4f}", flush=True)


def train_vae_patch_supervised(
    model,
    train_loader,      # yields (data, class_idx)
    test_loader,       # yields (data, class_idx)
    expert_protos,     # dict: expert_protos[class_idx][proto_idx] = numpy array (C, T)
    device,
    log=print,
    num_epochs=300,
    beta=1.0,           # VAE KL weight
    gamma=0.1,   # anchor loss weight
    cls_weight=0.1,     # patch classification loss weight
    anchor_warmup=20    # epochs before anchor loss
):
    # Losses
    recon_loss_fn = nn.MSELoss(reduction='sum')
    cls_loss_fn   = nn.CrossEntropyLoss()

    # Determine window size and number of classes
    st_cls = next(iter(k for k in expert_protos.keys() if expert_protos[k]))
    st_proto = next(iter(expert_protos[st_cls].values()))
    proto_window = st_proto.shape[1]
    num_classes = len(expert_protos)

    # Patch classification head
    # We'll infer latent channels C' from a dummy pass
    model.to(device)
    model.eval()
    with torch.no_grad():
        sample_x, _ = next(iter(train_loader))
        sample_x = sample_x.to(device)
        _, mu, _ = model(sample_x[:1])        # mu: (1, C', T')
        Cprime, Tprime = mu.shape[1], mu.shape[2]
    patch_dim = Cprime * proto_window

    patch_classifier = nn.Linear(patch_dim, num_classes).to(device)

    # Optimizer includes both model and classifier
    optimizer = optim.Adam(
        list(model.parameters()) + list(patch_classifier.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Precompute expert anchors (patch-level means)
    def compute_anchor_means():
        model.eval()
        means = {}
        with torch.no_grad():
            for cls, proto_dict in expert_protos.items():
                all_patches = []
                for proto in proto_dict.values():
                    x = torch.tensor(proto, dtype=torch.float32, device=device).unsqueeze(0)
                    _, mu, _ = model(x)  # (1, C', T')
                    pts = mu.unfold(2, proto_window, 1)  # (1, C', np, W)
                    _, c, np, w = pts.shape
                    pts = pts.permute(0,2,1,3).reshape(np, c*w)
                    all_patches.append(pts)
                means[cls] = torch.cat(all_patches,0).mean(dim=0)
        return means

    for epoch in range(1, num_epochs+1):
        anchors = None

        # Recompute anchors after warmup
        if epoch > anchor_warmup:
            anchors = compute_anchor_means()

        model.train()
        patch_classifier.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # Forward VAE
            recon_x, mu, logvar = model(x)

            # VAE losses
            recon_loss = recon_loss_fn(recon_x, x)
            kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            # Extract patches and flatten
            patches = mu.unfold(2, proto_window, 1)  # (B, C', np, W)
            b, c, np, w = patches.shape
            patches = patches.permute(0,2,1,3).reshape(b, np, c*w)

            # Supervised patch classification using mean-patch
            z_mean = patches.mean(dim=1)  # (B, patch_dim)
            logits = patch_classifier(z_mean)  # (B, num_classes)
            cls_loss = cls_loss_fn(logits, y)
            loss = loss + cls_weight * cls_loss

            # Anchor loss
            if anchors is not None:
                anchor_loss = 0.0
                for i in range(b):
                    cls_i = int(y[i].item())
                    anchor_vec = anchors[cls_i]  # (patch_dim,)
                    # Compute min-distance across patches
                    d2 = torch.sum((patches[i] - anchor_vec.unsqueeze(0))**2, dim=1)
                    anchor_loss += d2.min()
                loss = loss + gamma * anchor_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Logging every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            patch_classifier.eval()
            val_recon = val_kl = val_cls = val_anchor = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    recon_x, mu, logvar = model(x)
                    recon_loss = recon_loss_fn(recon_x, x)
                    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss

                    # patch classification val loss
                    patches = mu.unfold(2, proto_window, 1)
                    b,c,np,w = patches.shape
                    patches = patches.permute(0,2,1,3).reshape(b, np, c*w)
                    z_mean = patches.mean(dim=1)
                    logits = patch_classifier(z_mean)
                    cls_val = cls_loss_fn(logits, y)
                    loss = loss + cls_weight * cls_val

                    anchors = compute_anchor_means()

                    # anchor val loss if warmed up
                    anchor_v = 0.0
                    for i in range(b):
                        cls_i = int(y[i].item())
                        anchor_vec = anchors[cls_i]
                        d2 = torch.sum((patches[i] - anchor_vec)**2, dim=1)
                        anchor_v += d2.min()

                    if epoch > anchor_warmup:
                        loss = loss + gamma * anchor_v
                    val_anchor += anchor_v.item()

                    val_anchor += anchor_v.item()
                    val_recon += recon_loss.item()
                    val_kl    += kl_loss.item()
                    val_cls   += cls_val.item()

            n = len(test_loader.dataset)
            log(f"Epoch {epoch}/{num_epochs}", flush=True)
            log(f" Recon: {val_recon/n:.4f}, KL: {val_kl/n:.4f}", flush=True)
            log(f" Cls:   {val_cls/n:.4f}, Anchor: {val_anchor/n:.4f}", flush=True)
            # log(f" Total: {loss.item()/n:.4f}", flush=True)


def train_vae_patch_windowed(
    model,
    train_loader,      # yields (data, class_idx)
    test_loader,       # yields (data, class_idx)
    expert_protos,     # dict: expert_protos[class_idx][proto_idx] = numpy array (C, T)
    device,
    log=print,
    num_epochs=300,
    beta=1.0,           # VAE KL weight
    gamma_anchor=0.1,   # anchor loss weight
    cls_weight=0.1,     # patch classification loss weight
    anchor_warmup=20    # epochs before anchor loss
):
    recon_loss_fn = nn.MSELoss(reduction='sum')
    cls_loss_fn   = nn.CrossEntropyLoss()

    # Determine window size and number of classes
    st_cls = next(iter(expert_protos))
    st_proto = next(iter(expert_protos[st_cls].values()))
    proto_window = st_proto.shape[1]  # length in time
    num_classes = len(expert_protos)

    # Patch classification head (dim will be C'*proto_window)
    model.to(device)
    model.eval()
    with torch.no_grad():
        sample_x, _ = next(iter(train_loader))
        sample_x = sample_x.to(device)
        # sample a window
        x0 = sample_x[0:1, :, :proto_window]
        _, mu0, _ = model(x0)
        Cprime, Tprime = mu0.shape[1], mu0.shape[2]
    patch_dim = Cprime * Tprime
    patch_classifier = nn.Linear(patch_dim, num_classes).to(device)

    # optimizer includes both VAE and classifier params
    optimizer = optim.Adam(
        list(model.parameters()) + list(patch_classifier.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    def compute_anchor_means():
        model.eval()
        means = {}
        with torch.no_grad():
            for cls, proto_dict in expert_protos.items():
                zflats = []
                for proto in proto_dict.values():
                    x = torch.tensor(proto, dtype=torch.float32, device=device).unsqueeze(0)
                    # slice to window if longer
                    if x.shape[2] > proto_window:
                        x = x[:, :, :proto_window]
                    _, mu, _ = model(x)
                    zflat = mu.view(1, -1)  # (1, C'*proto_window)
                    zflats.append(zflat)
                if zflats:
                    means[cls] = torch.cat(zflats, dim=0).mean(dim=0)
        return means

    for epoch in range(1, num_epochs+1):
        anchors = None
        # recompute anchors after warmup
        if epoch > anchor_warmup:
            anchors = compute_anchor_means()

        model.train()
        patch_classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B, C, T = x.shape

            # sample random windows of length proto_window
            starts = torch.randint(0, T - proto_window + 1, (B,), device=device)
            x_win = torch.stack([x[i,:,starts[i]:starts[i]+proto_window] for i in range(B)], dim=0)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x_win)

            # VAE loss on windows
            recon_loss = recon_loss_fn(recon_x, x_win)
            kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            # flatten latent window
            z_flat = mu.view(B, -1)  # (B, patch_dim)

            # supervised patch classification
            logits = patch_classifier(z_flat)
            cls_loss = cls_loss_fn(logits, y)
            loss = loss + cls_weight * cls_loss

            # anchor val loss based on minimal distance of patches
            if anchors is not None:
                patches = mu.unfold(2, proto_window//(x.shape[2]//mu.shape[2]), 1)
                b, c, npatches, w = patches.shape
                patches = patches.permute(0,2,1,3).reshape(b, npatches, c*w)

                anchor_v = 0.0
                for i in range(b):
                    cls_i = int(y[i].item())
                    anchor_vec = anchors.get(cls_i)
                    if anchor_vec is None:
                        continue
                    diffs = patches[i] - anchor_vec.unsqueeze(0)
                    d2 = torch.sum(diffs * diffs, dim=1)
                    anchor_v += d2.min()
                loss = loss + gamma_anchor * anchor_v
                val_anchor += anchor_v.item()

            optimizer.step()

        scheduler.step()

        # logging
        if epoch % 10 == 0:
            model.eval(); patch_classifier.eval()
            tot_recon=tot_kl=tot_cls=tot_anchor=0.0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                B, C, T = x.shape
                # use first window for validation
                x_win = x[:,:, :proto_window]
                recon_x, mu, logvar = model(x_win)
                recon_loss = recon_loss_fn(recon_x, x_win)
                kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                z_flat = mu.view(B, -1)
                logits = patch_classifier(z_flat)
                cls_loss = cls_loss_fn(logits, y)
                anchor_l = 0.0
                anchors = compute_anchor_means()
                for i in range(B):
                    anchor_vec = anchors.get(int(y[i].item()))
                    if anchor_vec is None:
                        continue
                    diff = z_flat[i] - anchor_vec
                    anchor_l += torch.sum(diff * diff)
                tot_recon  += recon_loss.item()
                tot_kl     += kl_loss.item()
                tot_cls    += cls_loss.item()
                if epoch > anchor_warmup:
                    tot_anchor += anchor_l.item()
                tot_anchor += anchor_l.item()
            n = len(test_loader.dataset)
            log(f"Epoch {epoch}/{num_epochs}", flush=True)
            log(f" Recon:{tot_recon/n:.4f} KL:{tot_kl/n:.4f} Cls:{tot_cls/n:.4f} Anc:{tot_anchor/n:.4f}", flush=True)
            log(f" Total:{(tot_recon+tot_kl+tot_cls+tot_anchor)/n:.4f}", flush=True)
