import torch
import torchvision
import sys
PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import otfusion.cifar.train as cifar_train

def get_inp_tar(dataset):
    return dataset.data.view(dataset.data.shape[0], -1).float(), dataset.targets

def get_mnist_dataset(root, is_train, to_download, return_tensor=False):
    mnist = torchvision.datasets.MNIST(root, train=is_train, download=to_download,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                       # only 1 channel
                       (0.1307,), (0.3081,))
                 ]))

    if not return_tensor:
        return mnist
    else:
        return get_inp_tar(mnist)

class Dummy_Dataset( torch.utils.data.Dataset):
    def __init__(self, shape=(1,28,28), length=10000) -> None:
        super().__init__()
        self.shape = shape
        self.length = length
    def __getitem__(self, index):
        return torch.rand(self.shape), 0
    def __len__(self):
        return self.length


def get_dataloader(args, batch_size_train=1, batch_size_test=1, enable_shuffle=True, dummy_shape=(1,28,28), root_dir="."):

    # torch.manual_seed(args.seed)
    bsz = (batch_size_train, batch_size_test)


    if args.dataset.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(f'{root_dir}/files/', train=True, download=args.to_download,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           # only 1 channel
                                           (0.1307,), (0.3081,))
                                     ])),
          batch_size=bsz[0], shuffle=enable_shuffle
        )


        test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST(f'{root_dir}/files/', train=False, download=args.to_download,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                     ])),
          batch_size=bsz[1], shuffle=enable_shuffle
        )
    elif args.dataset.lower() == 'cifar10':
        if hasattr(args, 'cifar_style_data') and args.cifar_style_data:
            print("args.cifar_style_data")
            train_loader, test_loader = cifar_train.get_dataset(args.config)
        else:

            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(f'{root_dir}/data/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])),
                batch_size=bsz[0], shuffle=enable_shuffle
            )

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10(f'{root_dir}/data/', train=False, download=args.to_download,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])),
                batch_size=bsz[1], shuffle=enable_shuffle
            )

    elif args.dataset.lower() == 'dummy':
        train_loader = torch.utils.data.DataLoader(
          Dummy_Dataset(dummy_shape), batch_size=bsz[0]
        )   
        test_loader = torch.utils.data.DataLoader(
          Dummy_Dataset(dummy_shape), batch_size=bsz[1]
        )

    return train_loader, test_loader
