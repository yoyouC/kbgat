import torch
# from torchviz import make_dot, make_dot_from_trace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
from random import shuffle


CUDA = torch.cuda.is_available()


def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()


def transe_score(h, r, t, h_a=None, t_a=None, p_norm=1):
    return (h + r - t).norm(p_norm, dim=1)

def distmult_score(h, r, t, h_a=None, t_a=None, p_norm=1):
    return - h * r * t

def hier_score(h, r, t, h_a, t_a, p_norm=1):
    return t_a - h_a - (h - t).norm(p_norm, dim=1)

def simi_score(h, r, t, h_a, t_a, p_norm=1):
    return t_a - h_a + (h - t).norm(p_norm, dim=1)

def regularization(h, r, t, regul_rate):
    return ((torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3) * regul_rate

class BatchCategoryDataset():

    class CategoryDataset():
        def __init__(self, data):
            self.data = data
            shuffle(self.data)
            self.index = -1
        
        def __next__(self):
            self.index += 1
            return self.data[self.index]
        
        def __len__(self):
            return len(self.data) - (self.index + 1)

        def reset(self):
            self.index = -1

    def __init__(self, triplets, categories, batch_size):
        self.batch_size = batch_size

        self.cate_datasets = []
        for _, category in categories.items():
            category_triplets = self._filter_by_category(triplets, category)
            self.cate_datasets.append(self.CategoryDataset(category_triplets))
        
        self.count = 0
        self.curr_dataset = None
        self.dataset_count = 0
    
    def _filter_by_category(self, triplets, category):
        ret = []
        for t in triplets:
            _, r, _ = t
            if r in category:
                ret.append(t)
        return ret

        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count % self.batch_size == 0:
            self._sample_catagory()
        
        if not self.curr_dataset:
            self.reset()
            return None
        
        self.count += 1
        
        h, r, t = next(self.curr_dataset)
        return h, r, t

    def _sample_catagory(self):
        remains = []

        for dataset in self.cate_datasets:
            remains.append(len(dataset) // self.batch_size)
        remains = np.cumsum(remains)

        if remains[-1] == 0:
            self.curr_dataset = None
            return

        randint = random.randint(1, remains[-1])
        for i, num_remain in enumerate(remains):
            if randint <= num_remain:
                self.curr_dataset = self.cate_datasets[i]
                return

    def reset(self):
        for dataset in self.cate_datasets:
            dataset.reset()
    
    def max_next(self):
        i = self.dataset_count

        if i == len(self.cate_datasets):
            self.dataset_count = 0
            return None
        else:
            self.dataset_count += 1
            return self.cate_datasets[i].data
