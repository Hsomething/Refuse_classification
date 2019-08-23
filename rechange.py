import tensorflow as tf
from tensorflow.python.framework import graph_util
'''
cpktLogFileName = r'model/checkpoint'  #cpkt 文件路径
with open(cpktLogFileName, 'r') as f:
    #权重节点往往会保留多个epoch的数据，此处获取最后的权重数据      
    cpktFileName = f.readline().split('"')[1]
    print(cpktFileName)

h5FileName = r'model/model.h5'

reader = tf.train.NewCheckpointReader(cpktFileName)
f = h5py.File(h5FileName, 'w')
t_g = None
for key in sorted(reader.get_variable_to_shape_map()):
   # 权重名称需根据自己网络名称自行修改
   if key.endswith('w') or key.endswith('biases'):
        keySplits = key.split(r'/')
        keyDict = keySplits[1] + '/' + keySplits[1] + '/' + keySplits[2]
        f[keyDict] = reader.get_tensor(key)

import keras_segmentation
model = keras_segmentation.models.segnet.mobilenet_segnet(n_classes=2, input_height=224, input_width=224)
model.load_weights('model/')
model.save('model.h5')
'''
def freeze_graph(input_checkpoint, output_graph):
    '''

    :param input_checkpoint: xxx.ckpt(千万不要加后面的xxx.ckpt.data这种，到ckpt就行了!)
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "output" # 模型输入节点，根据情况自定义
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
input_checkpoint = 'model/model.ckpt'
out_graph = 'froze_model.pb'
freeze_graph(input_checkpoint, out_graph)
