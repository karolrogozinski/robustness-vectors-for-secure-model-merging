import os
import torch
import torchvision.datasets as datasets

class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

class ImageNet100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('./data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        location = './data'
        traindir = os.path.join(location, 'ImageNet100', 'train')
        valdir = os.path.join(location, 'ImageNet100', 'val')

        self.train_dataset = ImageFolderDataset(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]

        labels = {
                "n01968897": "chambered nautilus",
                "n01770081": "harvestman",
                "n01818515": "macaw",
                "n02011460": "bittern",
                "n01496331": "electric ray",
                "n01847000": "drake",
                "n01687978": "agama",
                "n01740131": "night snake",
                "n01537544": "indigo bunting",
                "n01491361": "tiger shark",
                "n02007558": "flamingo",
                "n01735189": "garter snake",
                "n01630670": "common newt",
                "n01440764": "tench",
                "n01819313": "sulphur-crested cockatoo",
                "n02002556": "white stork",
                "n01667778": "terrapin",
                "n01755581": "diamondback",
                "n01924916": "flatworm",
                "n01751748": "sea snake",
                "n01984695": "spiny lobster",
                "n01729977": "green snake",
                "n01614925": "bald eagle",
                "n01608432": "kite",
                "n01443537": "goldfish",
                "n01770393": "scorpion",
                "n01855672": "goose",
                "n01560419": "bulbul",
                "n01592084": "chickadee",
                "n01914609": "sea anemone",
                "n01582220": "magpie",
                "n01667114": "mud turtle",
                "n01985128": "crayfish",
                "n01820546": "lorikeet",
                "n01773797": "garden spider",
                "n02006656": "spoonbill",
                "n01986214": "hermit crab",
                "n01484850": "great white shark",
                "n01749939": "green mamba",
                "n01828970": "bee eater",
                "n02018795": "bustard",
                "n01695060": "Komodo dragon",
                "n01729322": "hognose snake",
                "n01677366": "common iguana",
                "n01734418": "king snake",
                "n01843383": "toucan",
                "n01806143": "peacock",
                "n01773549": "barn spider",
                "n01775062": "wolf spider",
                "n01728572": "thunder snake",
                "n01601694": "water ouzel",
                "n01978287": "Dungeness crab",
                "n01930112": "nematode",
                "n01739381": "vine snake",
                "n01883070": "wombat",
                "n01774384": "black widow",
                "n02037110": "oystercatcher",
                "n01795545": "black grouse",
                "n02027492": "red-backed sandpiper",
                "n01531178": "goldfinch",
                "n01944390": "snail",
                "n01494475": "hammerhead",
                "n01632458": "spotted salamander",
                "n01698640": "American alligator",
                "n01675722": "banded gecko",
                "n01877812": "wallaby",
                "n01622779": "great grey owl",
                "n01910747": "jellyfish",
                "n01860187": "black swan",
                "n01796340": "ptarmigan",
                "n01833805": "hummingbird",
                "n01685808": "whiptail",
                "n01756291": "sidewinder",
                "n01514859": "hen",
                "n01753488": "horned viper",
                "n02058221": "albatross",
                "n01632777": "axolotl",
                "n01644900": "tailed frog",
                "n02018207": "American coot",
                "n01664065": "loggerhead",
                "n02028035": "redshank",
                "n02012849": "crane",
                "n01776313": "tick",
                "n02077923": "sea lion",
                "n01774750": "tarantula",
                "n01742172": "boa constrictor",
                "n01943899": "conch",
                "n01798484": "prairie chicken",
                "n02051845": "pelican",
                "n01824575": "coucal",
                "n02013706": "limpkin",
                "n01955084": "chiton",
                "n01773157": "black and gold garden spider",
                "n01665541": "leatherback turtle",
                "n01498041": "stingray",
                "n01978455": "rock crab",
                "n01693334": "green lizard",
                "n01950731": "sea slug",
                "n01829413": "hornbill",
                "n01514668": "cock"
            }
        self.classnames = [labels[class_name] for class_name in self.classnames]