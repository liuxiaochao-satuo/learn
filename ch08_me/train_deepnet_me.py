import sys, os

from ch08.train_deepnet import network, t_test, t_train, x_test, x_train
from common import trainer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet_me import DeepConvNet
from common_me.trainer_me import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params('params_me.pkl')
print('Saved Network Parameters!')