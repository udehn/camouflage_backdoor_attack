import torch
import torch.nn as nn
import copy

import sys

# sys.path.insert(0, "../..")
from utils.util import progress_bar
from trainModel import evaluate

def fp(netC, test_dl, test_dl_bd, loss_fn):
    # Prepare arguments
    num_classes = 10
    input_height = 224
    input_width = 224
    input_channel = 3

    # Load models
    device = torch.device("cuda")
    netC.to(device)
    netC.eval()
    netC.requires_grad_(False)

    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _, __) in enumerate(test_dl):
        inputs = inputs.to(device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    # acc_clean = []
    # acc_bd = []

    outfile = "./logs/fine_pruning/{}-fine_pruning-results.txt".format('imagenet10')

    with open(outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, num_classes)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].bn2.running_mean = netC.layer4[1].bn2.running_mean[pruning_mask]
                    module[1].bn2.running_var = netC.layer4[1].bn2.running_var[pruning_mask]
                    module[1].bn2.weight.data = netC.layer4[1].bn2.weight.data[pruning_mask]
                    module[1].bn2.bias.data = netC.layer4[1].bn2.bias.data[pruning_mask]

                    module[1].ind = pruning_mask

                elif "linear" == name:

                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(device)

            test_orig_loader_accuracy, test_orig_loader_loss = evaluate(net_pruned, test_dl, loss_fn=loss_fn)

            test_trig_loader_accuracy, test_trig_loader_loss = evaluate(net_pruned, test_dl_bd, loss_fn=loss_fn)

            clean = test_orig_loader_accuracy
            bd = test_trig_loader_accuracy

            outs.write("%d %0.3f %0.3f\n" % (index, clean, bd))