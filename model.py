import torch
import torch.nn as nn
import torch.nn.functional as F

from receptive_field import compute_proto_layer_rf_info_v2

import itertools

class ProtoTSNet(nn.Module):

    def __init__(self, cnn_base, num_features, ts_sample_len, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log'):

        super(ProtoTSNet, self).__init__()
        self.features = cnn_base
        self.num_features = num_features
        self.ts_sample_len = ts_sample_len

        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        # Here we are initializing the class identities of the prototypes
        # Without domain specific knowledge we allocate the same number of
        # prototypes for each class
        assert(self.num_prototypes % self.num_classes == 0)

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.add_on_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=self.prototype_shape[1],
                out_channels=self.prototype_shape[1],
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
        )

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )

        # do not make this just a tensor, since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        # the feature input to prototype layer
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def _l2_convolution(self, x):
        # apply self.prototype_vectors as l2-convolution filters on input x
        x2 = x ** 2
        x2_patch_sum = F.conv1d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1)
        p2_reshape = p2.view(-1, 1)

        xp = F.conv1d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        # global min pooling
        min_distances = -F.max_pool1d(-distances,
                                      kernel_size=(distances.size()[2],))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def set_last_layer_incorrect_connection(self, incorrect_class_connection):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in itertools.chain(self.features.modules(), self.add_on_layers.modules()):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_class_connection=-0.5)


def construct_PPNet(cnn_base, num_features, ts_sample_len,
                    prototype_shape, num_classes,
                    prototype_activation_function='log'):
    layer_filter_sizes, layer_strides, layer_paddings = [], [], []

    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=ts_sample_len,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )
    return ProtoTSNet(
        cnn_base=cnn_base,
        num_features=num_features,
        ts_sample_len=ts_sample_len,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
    )
