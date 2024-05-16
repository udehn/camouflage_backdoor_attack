import os
import numpy as np
import cv2
import tensorflow as tf
from torchvision import transforms

from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ScalingCamouflage(object):

    def __init__(self, sourceImg=None, targetImg=None, **kwargs):

        self.sourceImg = sourceImg
        self.targetImg = targetImg

        self.params = {}
        keys = self.params.keys()

        for key, value in kwargs.items():
            self.params[key] = value


    def estimateConvertMatrix(self, inSize, outSize):
        inputDummyImg = (self.params['img_factor'] * np.eye(inSize)).astype('uint8')
        outputDummyImg = self._resize(inputDummyImg, outShape=(inSize, outSize))

        convertMatrix = (outputDummyImg[:, :, 0] / (np.sum(outputDummyImg[:, :, 0], axis=1)).reshape(outSize, 1))

        return convertMatrix

    def _resize(self, inputImg, outShape=(0, 0)):
        func = self.params['func']
        interpolation = self.params['interpolation']

        if func is Image.Image.resize:
            inputImg = Image.fromarray(inputImg)

        if func is transforms.Resize:
            inputImg = Image.fromarray(inputImg)

        if func is cv2.resize:
            outputImg = func(inputImg, outShape, interpolation=interpolation)
        elif func is Image.Image.resize:
            outputImg = func(inputImg, outShape, interpolation)
            outputImg = np.array(outputImg)
        else:
            outputImg = func(outShape[::-1], interpolation)(inputImg)
            outputImg = np.array(outputImg)

        if len(outputImg.shape) == 2:
            outputImg = outputImg[:, :, np.newaxis]

        return np.array(outputImg)

    def _getPerturbationGPU(self, convertMatrixL, convertMatrixR, source, target):

        penalty_factor = self.params['penalty']

        p, q, c = source.shape
        a, b, c = target.shape

        convertMatrixL = tf.constant(convertMatrixL, dtype=tf.float32)
        convertMatrixR = tf.constant(convertMatrixR, dtype=tf.float32)

        modifier_init = np.zeros(source.shape)

        source = tf.constant(source, dtype=tf.float32)
        target = tf.constant(target, dtype=tf.float32)

        modifier = tf.Variable(modifier_init, dtype=tf.float32)
        modifier_init = None

        attack = (tf.tanh(modifier) + 1) * 0.5

        x = tf.reshape(attack, [p, -1])
        x = tf.matmul(convertMatrixL, x)
        x = tf.reshape(x, [-1, q, c])
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [q, -1])
        x = tf.matmul(convertMatrixR, x)
        x = tf.reshape(x, [-1, a, c])
        output = tf.transpose(x, [1, 0, 2])

        delta_1 = attack - source
        delta_2 = output - target

        obj1 = tf.reduce_sum(tf.square(delta_1)) / (p * q)
        obj2 = penalty_factor * tf.reduce_sum(tf.square(delta_2)) / (a * b)

        obj = obj1 + obj2

        max_iteration = 3000
        with tf.compat.v1.Session() as sess:
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
            op = optimizer.minimize(obj, var_list=[modifier])
            sess.run(tf.compat.v1.global_variables_initializer())
            prev = np.inf
            for i in range(max_iteration):
                _, obj_value = sess.run([op, obj])
                if i % 1000 == 0:
                    print(obj_value)
                    if obj_value > prev * 0.999:
                        break
                    prev = obj_value
            attack_opt = attack.eval()
            # print("Obj1:", obj1.eval(), ", Obj2:", obj2.eval())
        return attack_opt

    def attack(self):

        sourceImg = self.sourceImg
        targetImg = self.targetImg

        sourceHeight, sourceWidth, sourceChannel = sourceImg.shape
        targetHeight, targetWidth, targetChannel = targetImg.shape

        convertMatrixL = self.estimateConvertMatrix(sourceHeight, targetHeight)
        convertMatrixR = self.estimateConvertMatrix(sourceWidth, targetWidth)
        img_factor = self.params['img_factor']
        sourceImg = sourceImg / img_factor
        targetImg = targetImg / img_factor

        source = sourceImg
        target = targetImg
        # self.info()
        attackImg = self._getPerturbationGPU(convertMatrixL,
                                             convertMatrixR,
                                             source, target)

        return np.uint8(attackImg * img_factor)

    def info(self):
        func_name = str(self.params['func'])
        inter_name = str(self.params['interpolation'])
        sourceShape = (self.sourceImg.shape[1],
                       self.sourceImg.shape[0],
                       self.sourceImg.shape[2])
        targetShape = (self.targetImg.shape[1],
                       self.targetImg.shape[0],
                       self.targetImg.shape[2])

        print('Source image size: %s' % str(sourceShape))
        print('Target image size: %s' % str(targetShape))
        print()
        print('Resize method: %s' % func_name)
        print('interpolation: %s' % inter_name)


def write_image(arr, path):
   arr = arr.astype(dtype='uint8')
   img = Image.fromarray(arr, 'RGB')
   img.save(path)


def generateOneAttackImgs(sourceImgPath, targetImgPath, attackImgPath, ind):

    targetImg = Image.open(targetImgPath)
    targetImg = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(targetImg)

    mask = Image.new("L", targetImg.size, 0)
    draw = ImageDraw.Draw(mask)

    # draw.ellipse((10, 60, 140, 160), fill=185)
    # draw.rectangle([25,25,50,50])
    # draw.rectangle([50,50,175,175], fill=75)
    draw.rectangle([224 - 10, 224 - 10, 224, 224], fill=int(1 * 255))
    # draw.rectangle([50,50,175,175], width=20)

    x, y = targetImg.size
    img2 = Image.open("./data/textures/2_leaf.jpg").resize((x, y))

    img3 = Image.composite(img2, targetImg, mask=mask)


    # sourceImg = Image.open(sourceImgPath)
    # sourceImg = np.array(sourceImg)
    # targetImg = np.array(img3)

    sourceImg = cv2.imread(sourceImgPath)
    sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB)
    targetImg = np.array(img3)


    ca1 = ScalingCamouflage(sourceImg,
                               targetImg,

                               func=cv2.resize,
                               #               func=transforms.Resize,
                               #               func=Image.Image.resize,

                               interpolation=cv2.INTER_LINEAR,
                               #               interpolation=InterpolationMode.BILINEAR,
                               #               interpolation=Image.BILINEAR,

                               penalty=0.1,
                               img_factor=255.)

    attackImg = ca1.attack()
    write_image(attackImg, attackImgPath)
