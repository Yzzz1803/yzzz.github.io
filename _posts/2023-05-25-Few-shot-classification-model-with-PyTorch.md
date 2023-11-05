---
title: Welcome
tags: PyTorch, Deep Learning
---

# Few-shot classification model with PyTorch

In 15 minutes and just a few lines of code, we are going to implement the [Prototypical Networks](https://arxiv.org/abs/1703.05175). It's the favorite method of many few-shot learning researchers (~2000 citations in 3 years), because 1) it works well, and 2) it's incredibly easy to grasp and to implement.

## Discovering Prototypical Networks
First, let's install the [tutorial GitHub repository](https://github.com/sicara/easy-few-shot-learning) and import some packages. If you're on Colab right now, you should also check that you're using a GPU (Edit > Notebook settings).
```python
!pip  install  easyfsl
```
```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average
```

Now we need a dataset. I suggest we use [Omniglot](https://github.com/brendenlake/omniglot), a popular MNIST-like benckmark for  few-shot classification. It contains 1623 characters from 50 different alphabets.  Each character has been written by 20 different people.
Also, It's a part of the `torchvision` package, so it's very convenient to download and work with.

```python
image_size = 28

# NB: background=True --> selects the train set, background=False --> selects the test set
# It's the nomenclature from the original paper, we just have to deal with it

train_set = Omniglot(
    root="./data",
    background=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)
test_set = Omniglot(
    root="./data",
    background=False,
    transform=transforms.Compose(
        [
            # Omniglot images have 1 channel, but our model will expect 3-channel images
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)
```

Let's taske some time to grasp what few-shot classification is. 
- Simply put, in a few-shot classification task, you have a labeled support set (which kind of acts like a catalog) and query set.
-  For each image of the query set, we want to predict a label from the labels present in the support set. 
- A few-shot classification model has to use the information from the support set in order to classify query images.

We say few-shot when the support set contains very few images for each label (typically less than 10).
The figure below shows a 3-way 2-shots classification task. "3-way" means "3 different classes" and "2-shots" means "2 examples per class". We expect a model that has never seen any Saint-Bernard, Pug or Labrador during its training to successfully predict the query labels. The support set is the only information that the model has regarding what a Saint-Bernard, a Pug or a Labrador can be.

![few-shot classification task](https://camo.githubusercontent.com/859fc228c6954111c4a4370dcf48c1ccce601651a5f882d0fcf9f00193982e68/68747470733a2f2f696d616765732e6374666173736574732e6e65742f62653034796c7038793071632f625a68626f7159586659655734493838786d4d4e762f37633565666463333638323036666561616430343563363734623163656439352f315f417465443079584c6b513142626a51544233597477672e706e673f666d3d77656270)

Most few-shot classification methods are metric-based. It works in two phases:

 1. They use a CNN to project both support and query images into a feature space.
 2. They classify query images by comparing them to support images.

If, in the feature space, an image is closer to pugs than it is to labradors and Saint-Bernards, we will guess that it's a pug.
From there, we have two challenges:
1. Find the good feature space. This is that convolutional networks are for. A CNN is basically a function that takes an image as input and outputs a representation (or embedding) of this image in a given feature space. The challenge here is to have a CNN that will project images of the same class into representations that are close to each other, even if it has not been trained an objects of this class.
2. Find a good way to compare the representations in the feature space. This is the job of Prototypical Networks.

![Prototypical classification](https://camo.githubusercontent.com/acbf519ebd077920caaf832452795fc97b8d80e9a727f67632c06578d10c517c/68747470733a2f2f696d616765732e6374666173736574732e6e65742f62653034796c7038793071632f34354d3955635570364b6e7a774461424865475a62372f62623264636461353934326565373332303630303132356163323331306166362f305f4d304753525a726938353966476f34382e706e673f666d3d77656270)

From the support set, Prototypical Networks compute a prototype for each class, which is the mean of all embeddings of support images from this class. Then, each query is simply classified as the nearest prototype in the feature space, with respect to euclidean distance.

In the code below(modified from [this](https://github.com/sicara/easy-few-shot-learning/blob/master/easyfsl/methods/prototypical_networks.py)), we simply define Prototypical Networks as a torch module, with a `forward()` method. You may notice 2 things.
1. We initiate `PrototypicalNetworks` with a backbone. This is the feature extractor we were talking about.

	Here, we use as backbone a ResNet18 pretrained on ImageNet, with its head chopped off and replaced by a `Flatten` layer. The output of the backbone, for an input image, will be a 512-dimensional feature vector.

2. The `forward` method doesn'y only take one input tensor, 
3. but  In order to predict the labels of query images, we also need support images and labels as input of the model.

```python
class PrototypicalNetworks(nn.Module):
	def __init__(self, backbone: nn.Module):
		super(PrototypicalNetwroks, self).__init__()
		self.backbone = backbone
	
	def forward(
		self,
		support_images:torch.Tensor,
		support_labels:torch.Tensor,
		query_images: torch.Tensor,
	) -> torch.Tensor:
	
	# Predict query labels using labeled support images.
	
	# Extract the features of support and query images
	z_support = self.backbone.forward(support_images)
	z_query = self.backbone.forward(query_images)

	# Infer the number of different classes from the labels of the support set
	n_way = len(torch.unique(support_images))
	# Prototype i is the mean of all instances of features corresponding to labels == i
	z_proto = torch.cat(
		[
			z_support[torch.nonzero(support_labels == labels)].mean(0)
			for label in range(n_way)
		]
	)

	# Compute the euclidean distance from queries to prototypes
	dists = torch.cdist(z_query, z_proto)

	# And here is the super complicated operation to transform those distances into classification scores!
	scores = -dists
	return scores

convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
print(convolutional_network)
```

Now we have a model~
Note that we used a pretrained feature extractor, so our model should already e up and running.
