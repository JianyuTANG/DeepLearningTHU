import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns
import os
from misc_functions import preprocess_image, recreate_image, save_image

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def tsne_visualize(model, validation_loader, device, png_name='digits_tsne-generated.png'):
    print("start visualization")
    model.train(False)
    features = []
    colors = []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            feature = model.get_feature(inputs)
            feature = feature.cpu().numpy()
            labels = labels.numpy()
            features += [feature[i] for i in range(feature.shape[0])]
            colors += [labels[i] for i in range(labels.shape[0])]
            # features.append(feature.cpu().numpy())
            # colors.append(labels.numpy())
    
    # stack every batch together
    features = np.vstack(features)
    colors = np.hstack(colors)

    features = TSNE(random_state=666).fit_transform(features)
    palette = np.array(sns.color_palette("hls", 20))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig(png_name, dpi=120)


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model.cpu()
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.convlayers = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.convlayers.append(m)
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.convlayers[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            # for index, layer in enumerate(self.model):
            #     # Forward pass layer by layer
            #     # x is not used after this point because it is only needed to trigger
            #     # the forward hook function
            #     x = layer(x)
            #     # Only need to forward until the selected layer is reached
            #     if index == self.selected_layer:
            #         # (forward hook function triggered)
            #         break
            x = self.model(x)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

