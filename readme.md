初学者的 TensorFlow 2.0 教程
在 Google Colab 中运行
在 GitHub 查看源代码
下载笔记本
此简短介绍使用 Keras 进行以下操作：

加载一个预构建的数据集。
构建对图像进行分类的神经网络机器学习模型。
训练此神经网络。
评估模型的准确率。
这是一个 Google Colaboratory 笔记本文件。 Python程序可以直接在浏览器中运行，这是学习 Tensorflow 的绝佳方式。想要学习该教程，请点击此页面顶部的按钮，在Google Colab中运行笔记本。

在 Colab中, 连接到Python运行环境： 在菜单条的右上方, 选择 CONNECT。
运行所有的代码块: 选择 Runtime > Run all。
设置 TensorFlow
首先将 TensorFlow 导入到您的程序：


import tensorflow as tf

2022-08-31 06:25:27.904924: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2022-08-31 06:25:28.630454: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvrtc.so.11.1: cannot open shared object file: No such file or directory
2022-08-31 06:25:28.630689: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvrtc.so.11.1: cannot open shared object file: No such file or directory
2022-08-31 06:25:28.630701: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
如果您在自己的开发环境而不是 Colab 中操作，请参阅设置 TensorFlow 以进行开发的安装指南。

注：如果您使用自己的开发环境，请确保您已升级到最新的 pip 以安装 TensorFlow 2 软件包。有关详情，请参阅安装指南。

加载数据集
加载并准备 MNIST 数据集。将样本数据从整数转换为浮点数：


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
构建机器学习模型
通过堆叠层来构建 tf.keras.Sequential 模型。


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
对于每个样本，模型都会返回一个包含 logits 或 log-odds 分数的向量，每个类一个。


predictions = model(x_train[:1]).numpy()
predictions

array([[-0.5549733 , -0.178904  ,  0.2521443 , -0.20637012,  0.24012126,
        -0.09939454, -0.43023387,  0.45244563,  0.37959167, -0.43027684]],
      dtype=float32)
tf.nn.softmax 函数将这些 logits 转换为每个类的概率：


tf.nn.softmax(predictions).numpy()

array([[0.0572833 , 0.08343588, 0.12839696, 0.0811754 , 0.12686247,
        0.09034067, 0.06489357, 0.15687165, 0.1458493 , 0.06489078]],
      dtype=float32)
注：可以将 tf.nn.softmax 烘焙到网络最后一层的激活函数中。虽然这可以使模型输出更易解释，但不建议使用这种方式，因为在使用 softmax 输出时不可能为所有模型提供精确且数值稳定的损失计算。

使用 losses.SparseCategoricalCrossentropy 为训练定义损失函数，它会接受 logits 向量和 True 索引，并为每个样本返回一个标量损失。


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
此损失等于 true 类的负对数概率：如果模型确定类正确，则损失为零。

这个未经训练的模型给出的概率接近随机（每个类为 1/10），因此初始损失应该接近 -tf.math.log(1/10) ~= 2.3。


loss_fn(y_train[:1], predictions).numpy()

2.4041677
在开始训练之前，使用 Keras Model.compile 配置和编译模型。将 optimizer 类设置为 adam，将 loss 设置为您之前定义的 loss_fn 函数，并通过将 metrics 参数设置为 accuracy 来指定要为模型评估的指标。


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
训练并评估模型
使用 Model.fit 方法调整您的模型参数并最小化损失：


model.fit(x_train, y_train, epochs=5)

Epoch 1/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2982 - accuracy: 0.9140
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1422 - accuracy: 0.9573
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1080 - accuracy: 0.9668
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0857 - accuracy: 0.9732
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0754 - accuracy: 0.9759
<keras.callbacks.History at 0x7fcad46da820>
Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上检查模型性能。


model.evaluate(x_test,  y_test, verbose=2)

313/313 - 1s - loss: 0.0674 - accuracy: 0.9791 - 588ms/epoch - 2ms/step
[0.06739047914743423, 0.9790999889373779]
现在，这个照片分类器的准确度已经达到 98%。想要了解更多，请阅读 TensorFlow 教程。

如果您想让模型返回概率，可以封装经过训练的模型，并将 softmax 附加到该模型：


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[5.22209795e-08, 2.41216696e-07, 1.94858385e-05, 6.03494642e-04,
        3.63586158e-12, 1.27487894e-07, 2.75295411e-12, 9.99375284e-01,
        3.34941291e-07, 1.01254307e-06],
       [1.27781030e-09, 5.08851197e-04, 9.99486089e-01, 1.27736268e-06,
        1.16675265e-17, 2.87630655e-06, 3.99693533e-07, 1.31204751e-14,
        4.28570189e-07, 8.51850329e-13],
       [8.57534303e-07, 9.99213696e-01, 5.82170615e-05, 2.93258381e-06,
        2.00496142e-05, 8.48563104e-06, 1.34474585e-05, 5.36364911e-04,
        1.44018544e-04, 2.03336299e-06],
       [9.97936487e-01, 3.06930929e-07, 9.99198644e-04, 2.78684138e-06,
        4.31515036e-06, 3.91961257e-05, 8.49796401e-04, 1.12621237e-04,
        4.11850197e-05, 1.40180309e-05],
       [4.24850878e-05, 1.55146904e-06, 2.20226084e-05, 7.31444686e-07,
        9.98439252e-01, 7.06452113e-07, 8.66830524e-05, 1.98422102e-04,
        2.75226976e-05, 1.18070992e-03]], dtype=float32)>
结论
恭喜！您已经利用 Keras API 借助预构建数据集训练了一个机器学习模型。

有关使用 Keras 的更多示例，请查阅教程。要详细了解如何使用 Keras 构建模型，请阅读指南。如果您想详细了解如何加载和准备数据，请参阅有关图像数据加载或 CSV 数据加载的教程。
