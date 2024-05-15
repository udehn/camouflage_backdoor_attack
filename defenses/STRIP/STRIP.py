import numpy as np
import cv2
import torch


def superimpose(background, overlay):
    added_image = cv2.addWeighted(background, 1, overlay, 1, 0)
    # return (added_image.reshape(32,32,3))
    return (added_image.reshape(3, 224, 224))


def entropyCal(background, n, model, x_train):
    device = torch.device('cuda')
    with torch.no_grad():
        entropy_sum = [0] * n
        x1_add = [0] * n
        # index_overlay = np.random.randint(40000,49999, size=n)
        index_overlay = np.random.randint(0, len(x_train), size=n)
        for x in range(n):
            x1_add[x] = (superimpose(background.numpy(), x_train[index_overlay[x]][0].numpy()))

        # py1_add = model.predict(np.array(x1_add))
        py1_add = model(torch.from_numpy(np.array(x1_add)).to(device))
        # py1_add = model(np.array(x1_add))

        EntropySum = -np.nansum(py1_add.cpu() * np.log2(py1_add.cpu()))
        return EntropySum

