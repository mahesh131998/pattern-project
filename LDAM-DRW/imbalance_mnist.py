import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, imbalance_dataset = False,
                 add_noise = False, noise_ratio = 10, asym_noise = False):
        folders = ['data', 'model', 'log']
        for folder in folders:
            path = os.path.join('./', folder)
            if not os.path.exists(path):
                os.makedirs(path)
        self.num_per_cls_dict = dict()
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        if add_noise:
            new_targets = self.create_noise(targets_np, noise_ratio, asym=asym_noise)
            self.targets = torch.FloatTensor(new_targets)
        if imbalance_dataset:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(targets_np, classes, img_num_list)
        img_num_list = self.get_img_num_per_cls(self.cls_num, "check", imb_factor)
        for the_class, the_img_num in zip(classes, img_num_list):
            self.num_per_cls_dict[the_class] = the_img_num

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        targets_np = np.array(self.targets, dtype=np.int64)
        if imb_type == 'check':
            img_num_per_cls = [int(np.where(targets_np == i)[0].shape[0]) for i in range(self.cls_num)]
        elif imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = int(np.where(targets_np == cls_idx)[0].shape[0]) * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(np.where(targets_np == cls_idx)[0].shape[0]))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(int(np.where(targets_np == cls_idx)[0].shape[0]) * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, targets_np, classes, img_num_per_cls):
        new_data = []
        new_targets = []
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        # print(new_targets)
        self.data = torch.from_numpy(new_data)
        self.targets = torch.FloatTensor(new_targets)
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def noise_helper(self, n_classes, current_class):
        if current_class < 0 or current_class >= n_classes:
            error_str = "class_ind must be within the range (0, nb_classes - 1)"
            raise ValueError(error_str)

        other_class_list = list(range(n_classes))
        other_class_list.remove(current_class)
        other_class = np.random.choice(other_class_list)
        return other_class

    def create_noise(self, y_tr, noise_ratio, asym = False):
        if noise_ratio > 0:
            dataset = 'mnist'
            noisy_y_tr = np.array(y_tr, copy=True)
            if asym:
                    data_file = "data/asym_%s_noisytrain_labels_%s.npy" % (dataset, noise_ratio)
            else:
                    data_file = "data/%s_noisytrain_labels_%s.npy" % (dataset, noise_ratio)
            if os.path.isfile(data_file):
                    y_tr_c = np.load(data_file)
            else:
                    if asym:
                        if dataset == 'mnist':
                            # 1 < - 5, 2 -> 4, 3 -> 7, 5 <-> 6, 8 -> 9
                            source_class = [5, 2, 3, 5, 6, 8]
                            target_class = [1, 4, 7, 6, 5, 9]
                        if dataset == 'mnist' :
                            for s, t in zip(source_class, target_class):
                                cls_idx = np.where(y_tr == s)[0]
                                n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)
                                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                                noisy_y_tr[noisy_sample_index] = t
                    else:
                        n_samples = noisy_y_tr.shape[0]
                        n_noisy = int(noise_ratio * n_samples / 100)
                        class_index = [np.where(y_tr == i)[0] for i in range(self.cls_num)]
                        class_noisy = int(n_noisy / 10)

                        noisy_idx = []
                        for d in range(self.cls_num):
                            noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                            noisy_idx.extend(noisy_class_index)

                        for i in noisy_idx:
                            noisy_y_tr[i] = self.noise_helper(n_classes=self.cls_num, current_class=y_tr[i])
                    np.save(data_file, noisy_y_tr)

            print("Print noisy label generation statistics:")
            count = 0
            for i in range(10):
                    n_noisy = np.sum(noisy_y_tr == i)
                    print("Noisy class %s, has %s samples." % (i, n_noisy))
                    count += n_noisy
            print(count)
            return noisy_y_tr

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    trainset = IMBALANCEMNIST(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()