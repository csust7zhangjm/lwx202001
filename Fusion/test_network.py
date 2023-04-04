import tensorflow as tf
from utils import weights_spectral_norm


class FusionNet():
    def feature_padding(self, x, kernel=3, stride=1, pad=1):
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left
        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        return x

    def vi_feature_extraction_network(self, vi_image, reader):
        with tf.compat.v1.variable_scope('vi_extraction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('Fusion_model/vi_extraction_network/conv1/w')))
                #weights = weights_spectral_norm(weights)
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                    'Fusion_model/vi_extraction_network/conv1/b')))
                input = self.feature_padding(vi_image, kernel=5, stride=1, pad=2)
                conv1 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
             
                conv1 = tf.nn.leaky_relu(conv1)
            block1_input = conv1
            print("block1_input shape: ", block1_input.get_shape().as_list())
            # state size: 16
            with tf.compat.v1.variable_scope('block1'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block1/conv1/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block1/conv1/b')))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block1/conv2/w')))
             
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block1/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block1/conv3/w')))
          
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block1/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                print("conv3 shape: ", conv3.get_shape().as_list())

                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block1/identity_conv/w')))
              
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID')

                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block1/conv4/w')))
            
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block1/conv4/b')))
                    block1_input=self.feature_padding(block1_input)
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block1_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.compat.v1.variable_scope('block2'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block2/conv1/w')))
              
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block2/conv1/b')))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block2/conv2/w')))
        
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block2/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block2/conv3/w')))
             
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block2/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block2/identity_conv/w')))
                   
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block2/conv4/w')))
            
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block2/conv4/b')))
                    block2_input=self.feature_padding(block2_input)
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block2_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.compat.v1.variable_scope('block3'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block3/conv1/w')))
                
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block3/conv1/b')))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block3/conv2/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block3/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block3/conv3/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block3/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block3/identity_conv/w')))
              
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/vi_extraction_network/block3/conv4/w')))
            
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/vi_extraction_network/block3/conv4/b')))
                    block3_input=self.feature_padding(block3_input)
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
         
                encoding_feature = block3_output
        return encoding_feature

    def ir_feature_extraction_network(self, ir_image, reader):
        with tf.compat.v1.variable_scope('ir_extraction_network'):
            with tf.compat.v1.variable_scope('conv1'):
                weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                    reader.get_tensor('Fusion_model/ir_extraction_network/conv1/w')))
                
                bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                    'Fusion_model/ir_extraction_network/conv1/b')))
                input = self.feature_padding(ir_image, kernel=5, stride=1, pad=2)
                conv1 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
               
                conv1 = tf.nn.leaky_relu(conv1)
            block1_input = conv1
            # state size: 16
            with tf.compat.v1.variable_scope('block1'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block1/conv1/w')))
                    
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block1/conv1/b')))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block1/conv2/w')))
              
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block1/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block1/conv3/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block1/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias

                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block1/identity_conv/w')))
                
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID')

                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block1/conv4/w')))
              
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block1/conv4/b')))
                    block1_input=self.feature_padding(block1_input)
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)

                block1_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.compat.v1.variable_scope('block2'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block2/conv1/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block2/conv1/b')))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block2/conv2/w')))
               
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block2/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block2/conv3/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block2/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block2/identity_conv/w')))
                 
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block2/conv4/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block2/conv4/b')))
                    block2_input=self.feature_padding(block2_input)
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block2_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.compat.v1.variable_scope('block3'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block3/conv1/w')))
                   
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block3/conv1/b')))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block3/conv2/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block3/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block3/conv3/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block3/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block3/identity_conv/w')))
                  
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/ir_extraction_network/block3/conv4/w')))
                   
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/ir_extraction_network/block3/conv4/b')))
                    block3_input=self.feature_padding(block3_input)
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
           
                encoding_feature = block3_output
        return encoding_feature

    def feature_reconstruction_network(self, feature, reader):
        with tf.compat.v1.variable_scope('reconstruction_network'):
            block1_input = feature
            with tf.compat.v1.variable_scope('block1'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block1/conv1/w')))
              
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block1/conv1/b')))
                    conv1 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block1/conv2/w')))
                   
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block1/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block1/conv3/w')))
                
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block1/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias

                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block1/identity_conv/w')))
                  
                    identity_conv = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block1/conv4/w')))
                
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block1/conv4/b')))
                    block1_input=self.feature_padding(block1_input)
                    conv4 = tf.nn.conv2d(block1_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)

                block1_output = tf.nn.elu(conv3 + identity_conv+conv4)
            block2_input = block1_output
            with tf.compat.v1.variable_scope('block2'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block2/conv1/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block2/conv1/b')))
                    conv1 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block2/conv2/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block2/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block2/conv3/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block2/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block2/identity_conv/w')))
                  
                    identity_conv = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID')
                    
                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block2/conv4/w')))
              
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block2/conv4/b')))
                    block2_input=self.feature_padding(block2_input)
                    conv4 = tf.nn.conv2d(block2_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)

                block2_output = tf.nn.elu(conv3 + identity_conv+conv4)
                block3_input = block2_output
            with tf.compat.v1.variable_scope('block3'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block3/conv1/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block3/conv1/b')))
                    conv1 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block3/conv2/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block3/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block3/conv3/w')))
                
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block3/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block3/identity_conv/w')))
                 
                    identity_conv = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID')

                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block3/conv4/w')))
                 
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block3/conv4/b')))
                    block3_input=self.feature_padding(block3_input)
                    conv4 = tf.nn.conv2d(block3_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block3_output = tf.nn.leaky_relu(conv3 + identity_conv+conv4)
                block4_input = block3_output
            with tf.compat.v1.variable_scope('block4'):
                with tf.compat.v1.variable_scope('conv1'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block4/conv1/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block4/conv1/b')))
                    conv1 = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv1 = tf.nn.leaky_relu(conv1)

                with tf.compat.v1.variable_scope('conv2'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block4/conv2/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block4/conv2/b')))
                    input = self.feature_padding(conv1)
                    conv2 = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv2 = tf.nn.leaky_relu(conv2)
                with tf.compat.v1.variable_scope('conv3'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block4/conv3/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block4/conv3/b')))
                    conv3 = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                with tf.compat.v1.variable_scope('identity_conv'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block4/identity_conv/w')))
                   
                    identity_conv = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='VALID')

                with tf.compat.v1.variable_scope('conv4'):
                    weights = tf.compat.v1.get_variable("w", initializer=tf.constant(
                        reader.get_tensor('Fusion_model/reconstruction_network/block4/conv4/w')))
                  
                    bias = tf.compat.v1.get_variable("b", initializer=tf.constant(reader.get_tensor(
                        'Fusion_model/reconstruction_network/block4/conv4/b')))
                    block4_input=self.feature_padding(block4_input)
                    conv4 = tf.nn.conv2d(block4_input, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
                    conv4 = tf.nn.leaky_relu(conv4)
                block4_output = tf.nn.tanh(conv3 + identity_conv+conv4)
        
                fusion_image = block4_output
        return fusion_image

    def Fusion_model(self, vi_image, ir_image, reader):
        with tf.compat.v1.variable_scope("Fusion_model"):
            vi_encoding_feature = self.vi_feature_extraction_network(vi_image, reader)
            ir_encoding_feature = self.ir_feature_extraction_network(ir_image, reader)
            feature = tf.concat([vi_encoding_feature, ir_encoding_feature], axis=-1)
            f_image = self.feature_reconstruction_network(feature, reader)
        return f_image, feature
