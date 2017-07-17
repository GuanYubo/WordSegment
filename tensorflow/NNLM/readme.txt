TextCNN: 用卷积神经网络进行文本分类，将词表示为分布式的词向量（一般为几十到几百维），用固定维度的向量来表示一个词。
对于一个含有若干词的句子，卷积神经网络采用如下方法处理：
1.将每个词换成相应的m维词向量，从而将一句话表示成一个n*m的特征矩阵（此处需指定句子长度n），即神经卷积神经网络只能处理句子的前n个词，句子中的词如果不到n个则用0来填充
2.通过不同尺寸的卷积核来对句子的特征矩阵进行卷积操作，之后经过最大池化后得到一个值
  每个卷积核对特征矩阵进行卷积并池化后得到一个标量，一般选择几个不同尺寸的卷积核，每种尺寸的卷积核也有多个，因此最后池化后的输出层有多个值（一般在100左右）
3.池化层的输出经过加权并由softmax多项式进行归一化，得到最后每一类的概率
关于以上步骤的细节请参考这篇paper：Convolutional Neural Networks for Sentence Classification

类：
TextCNN(sequence_length, num_classes, vocabulary, embedding_size=128,
filter_sizes=[1,2,3], num_filters=24, l2_reg_lambda=0.0)

参数:
	sequence_length		指定句子长度，即特征矩阵的第二维（不建议设为样本的最大句子长度，有个别句子可能过长，导致模型参数过多）
	num_classes			类别数
	vocab_size			字典中含有的词数
	vocabulary			句子向量化参考字典，如果不提供，fit_corpus方法拟合时将按照词频生成一个字典（用卡方筛选会导致模型学习效果极差，目前还未发现原因）
	embedding_size 		词向量空间的维度，即将一个词表示成多少维的向量，默认值128
	filter_sizes		卷积核尺寸的列表，默认值[1,2,3]
	num_filters			每个尺寸的卷积核数量，默认值24
	l2_reg_lambda		l2正则项的系数，用来防止模型过拟合，越大表示越强的约束（从结果来看效果不明显），默认是0.0


方法：
fit_corpus(corpus)
拟合样本
	corpus应为一维numpy数组或列表，每个元素为已经切好词的字符串，词与词之间用' '相隔
	返回一个2维矩阵，第一维是样本个数，第二维是句子长度（即sequence_length)

fit(X_trian, y_train, X_test=None, y_test=None, num_epochs=100,
	batch_size=512, keep_drop=0.5, save_path=None, verbose=False)
训练模型
	X_train		fit_corpus返回的2维矩阵
	y_train		标签列表或numpy数组
	X_test, y_test		测试样本，建议提供，防止模型过拟合
	num_epochs		如果训练一直没有满足迭代停止条件，将在进行num_epochs轮迭代后停止，默认值100
	batch_size		每轮迭代时会先将所有样本打散，然后将样本分成多批进行训练，batch_size是每批样本的个数，默认值512
	keep_drop		有池化层的输出到最后输出概率为全连接层，为防止过拟合，再没批迭代时需要随机忽略部分输出结果，keep_drop为保留的神经元比例，默认值0.5
	save_path		保存模型的路径, 会自动创建该文件夹并将模型相关的文件存在文件夹下
	verbose			训练过程是否输出每轮迭代结果（损失函数，在训练集上的准确度，在测试集上的准确度（如果提供了测试集的话））

predict(X_test)
预测样本标签
	X_test为经过fit_corpus拟合过的测试样本
	返回类别标签列表（一维numpy数组）

predict_proba(X_test)
预测样本概率
	X_test为经过fit_corpus拟合过的测试样本
	返回二维numpy数组，第一维为样本个数，第二维为样本属于每一类的概率

--------------------------------------------------------------------------------------------------------


类：
CNN(path):	继承自TextCNN，从保存模型文件的文件夹中读取信息并重建模型，重写了predict和predict_proba方法，直接对已切词但未经拟合的消息文本进行预测

方法：
predict(input_words)
预测消息标签
	input_words为字符串列表
	返回类别标签列表

predict_proba(input_words)
预测消息的类别概率
	输入同上，返回属于每一类的概率


------------------------------------------------------------------------------------------------------------------











