import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# [B, H, W, C]
class LeNet(Model):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(16, 5, activation="relu") # 16个5*5的卷积核, output(28, 28, 16)
        self.pool1 = MaxPool2D(2, 2, padding="SAME") # 步距为2的2*2最大池化核, out(14, 14, 16)
        self.conv2 = Conv2D(32, 5, activation="relu") # output(10, 10, 32)
        self.pool2 = MaxPool2D(2, 2, padding="SAME") # out(5, 5, 32)
        self.flatten = Flatten() # 展平处理
        self.fc1 = Dense(120) # 输出120维的向量
        self.fc2 = Dense(84)
        # self.fc3 = Dense(num_classes)
        self.fc3 = Dense(num_classes, activation="softmax") # logits:各种情况相加等于1，也通过设定from_logits=True 标志位将softmax 激活函数实现在损失函数中

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    input1 = tf.random.normal([8, 32, 32, 3])
    model = LeNet(10)
    output = model(input1)
    print(output)






