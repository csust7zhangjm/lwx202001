# import tensorflow as tf
import tensorflow.compat.v1 as tf
from utils import weights_spectral_norm


class FusionNet():
    def vi_feature_extraction_network(self, vi_image):
        with tf.variable_scope('vi_extraction_network'): #可以让变量有相同的命名 它返回的是一个用于定义创建variable(层)的op的上下文管理器
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("w", [5, 5, 1, 16],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                #weights = weights_spectral_norm(weights)
                bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(vi_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.leaky_relu(conv1)
            block1_input = conv1
            # state size: 16
            with tf.variable_scope('block1'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias


                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME')

                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                # block1 没有identity_conv
                block1_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.variable_scope('block2'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block2_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.variable_scope('block3'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                encoding_feature = block3_output
         

        return encoding_feature

    def ir_feature_extraction_network(self, ir_image):
        with tf.variable_scope('ir_extraction_network'):
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("w", [5, 5, 1, 16],
                                                    initializer=tf.truncated_normal_initializer(stddev=1e-3))
                #weights = weights_spectral_norm(weights)
                bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(ir_image, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                # conv1 = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
                conv1 = tf.nn.leaky_relu(conv1)
            block1_input = conv1
            # state size: 16
            with tf.variable_scope('block1'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias

                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME')

                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)

                block1_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.variable_scope('block2'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 16, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block2_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.variable_scope('block3'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 32, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                encoding_feature = block3_output
       
        return encoding_feature

    def feature_reconstruction_network(self, feature):
        with tf.variable_scope('reconstruction_network'):
            block1_input = feature
            with tf.variable_scope('block1'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 128, 128],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 128, 128],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 128, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 128, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME')

                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 128, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block1_output = tf.nn.elu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.variable_scope('block2'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 64, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 64, 64],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 64, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 64, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME')

                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 64, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)   
                block2_output = tf.nn.elu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.variable_scope('block3'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 32, 32],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [32], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 32, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 32, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME')

                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 32, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)   
                
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block4_input = block3_output
            with tf.variable_scope('block4'):
                with tf.variable_scope('conv1'):
                    weights = tf.get_variable("w", [1, 1, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.variable_scope('conv2'):
                    weights = tf.get_variable("w", [3, 3, 16, 16],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [16], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.variable_scope('conv3'):
                    weights = tf.get_variable("w", [1, 1, 16, 1],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                with tf.variable_scope('identity_conv'):
                    weights = tf.get_variable("w", [1, 1, 16, 1],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    identity_conv = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                
                with tf.variable_scope('conv4'):
                    weights = tf.get_variable("w", [3, 3, 16, 1],
                                                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    #weights = weights_spectral_norm(weights)
                    bias = tf.get_variable("b", [1], initializer=tf.constant_initializer(0.0))
                    conv4 = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
                    conv4 = tf.nn.leaky_relu(conv4)   
                
                block4_output = tf.nn.tanh(conv3 + identity_conv+conv4)
                fusion_image = block4_output
       
        return fusion_image

    def Fusion_model(self, vi_image, ir_image):
        with tf.variable_scope("Fusion_model"):
            vi_feature = self.vi_feature_extraction_network(vi_image)
            ir_feature = self.ir_feature_extraction_network(ir_image)
            feature = tf.concat([vi_feature, ir_feature], axis=-1)
            f_image = self.feature_reconstruction_network(feature)
        return f_image
