# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Type
from tensorflow.keras import layers,models

from tflite2xcore.xcore_schema import XCOREOpCodes, BuiltinOpCodes
from tflite2xcore.model_generation import Configuration

from . import (  # pylint: disable=unused-import
    test_output_tdnn,
)
from .. import IntegrationTestModelGenerator, TdnnTestRunner

        
#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class KeywordRecognizer(IntegrationTestModelGenerator):
    def _build_core_model(self) -> tf.keras.Model:
        """
        Build a model with 3 CNN layers and a final FC layer (integrated post-processing)
        :param model_params: parameters used in model architecture/building
        :param feature_params: parameters used in extracting features
        :return: model
        """
        model_seed= 1
        filter_CNN1_height= 1
        filter_CNN1_width= 21
        filter_CNN1_chans= 64
        strides_CNN1_height= 1
        strides_CNN1_width= 1
        max_pool_CNN1_height= 1
        max_pool_CNN1_width= 3
        max_pool_CNN1_stride_height= 1
        max_pool_CNN1_stride_width= 2
        filter_CNN2_height= 1
        filter_CNN2_width= 9
        filter_CNN2_chans= 32
        strides_CNN2_height= 1
        strides_CNN2_width= 1
        max_pool_CNN2_height= 1
        max_pool_CNN2_width= 2
        max_pool_CNN2_stride_height= 1
        max_pool_CNN2_stride_width= 2
        filter_CNN3_height= 1
        filter_CNN3_width= 6
        filter_CNN3_chans= 1
        strides_CNN3_height= 1
        strides_CNN3_width= 1
        label_count= 1
        kernel_init_mean= 0
        kernel_init_stddev= 0.01
     
        # input
        input_frequency_size = 8 
        input_time_size =  94 

        # model instantiation
        model = tf.keras.Sequential()

        #################################################### INPUT layer ###################################################
        #################### conv_model_v_24 -> in: [batch_size, 1, 94, 8]; out: [batch_size, 1, 94, 8] ####################
        # input layer (allows to call model.summary() before model.fit())
        INPUT = model.add(tf.keras.Input(shape=(1, input_time_size, input_frequency_size), name="INPUT_LAYER"))

        ###################################### CNN1 + MaxPool + Activation + BatchNorm #####################################
        #################### conv_model_v_24 -> in: [batch_size, 1, 94, 8]; out: [batch_size, 1, 36, 64] ###################
        # CNN1 params.
        filter_CNN1_height =  filter_CNN1_height 
        filter_CNN1_width =  filter_CNN1_width 
        filter_CNN1_chans =  filter_CNN1_chans 
        strides_CNN1_height =  strides_CNN1_height 
        strides_CNN1_width =  strides_CNN1_width 

        filter_CNN1_shape = (filter_CNN1_height, filter_CNN1_width)
        strides_CNN1_shape = (strides_CNN1_height, strides_CNN1_width)

        # CNN1 (conv_model_v_24 -> in: [batch_size, 1, 94, 8]; out: [batch_size, 1, 74, 64])
        CNN1 = model.add(
            tf.keras.layers.Conv2D(
                filter_CNN1_chans,
                filter_CNN1_shape,
                strides=strides_CNN1_shape,
                padding="valid",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean= kernel_init_mean ,
                    stddev= kernel_init_stddev ,
                    seed= model_seed ,
                ),
                name="CNN1",
            )
        )

        # MaxPool (conv_model_v_24 -> in: [batch_size, 1, 74, 64]; out: [batch_size, 1, 36, 64])
        MaxPool_CNN1 = model.add(
            tf.keras.layers.MaxPool2D(
                pool_size=( max_pool_CNN1_height ,  max_pool_CNN1_width ),
                strides=( max_pool_CNN1_stride_height ,  max_pool_CNN1_stride_width ),
                name="MAXPOOL_CNN1",
            )
        )

        # ReLU (conv_model_v_24 -> in: [batch_size, 1, 36, 64]; out: [batch_size, 1, 36, 64])
        ReLU_CNN1 = model.add(tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="RELU_CNN1"))

        # BatchNorm (conv_model_v_24 -> in: [batch_size, 1, 36, 64]; out: [batch_size, 1, 36, 64])
        # BatchNorm_CNN1 = model.add(tf.keras.layers.BatchNormalization(name="BATCHNORM_CNN1"))

        ###################################### CNN2 + MaxPool + Activation + BatchNorm #####################################
        ##################### conv_model_v_24 -> in: [batch_size, 1, 36, 64]; out: [batch_size, 1, 14, 32] #################
        # CNN2 params.
        filter_CNN2_height =  filter_CNN2_height 
        filter_CNN2_width =  filter_CNN2_width 
        filter_CNN2_chans =  filter_CNN2_chans 
        strides_CNN2_height =  strides_CNN2_height 
        strides_CNN2_width =  strides_CNN2_width 

        filter_CNN2_shape = (filter_CNN2_height, filter_CNN2_width)
        strides_CNN2_shape = (strides_CNN2_height, strides_CNN2_width)

        # CNN2 (conv_model_v_24 -> in: [batch_size, 1, 36, 64]; out: [batch_size, 1, 28, 32])
        CNN2 = model.add(
            tf.keras.layers.Conv2D(
                filter_CNN2_chans,
                filter_CNN2_shape,
                strides=strides_CNN2_shape,
                padding="valid",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean= kernel_init_mean ,
                    stddev= kernel_init_stddev ,
                    seed= model_seed ,
                ),
                name="CNN2",
            )
        )

        # MaxPool (conv_model_v_24 -> in: [batch_size, 1, 28, 32]; out: [batch_size, 1, 14, 32])
        MaxPool_CNN2 = model.add(
            tf.keras.layers.MaxPool2D(
                pool_size=( max_pool_CNN2_height ,  max_pool_CNN2_width ),
                strides=( max_pool_CNN2_stride_height ,  max_pool_CNN2_stride_width ),
                name="MAXPOOL_CNN2",
            )
        )

        # ReLU (conv_model_v_24 -> in: [batch_size, 1, 14, 32]; out: [batch_size, 1, 14, 32])
        ReLU_CNN2 = model.add(tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="RELU_CNN2"))

        # BatchNorm (conv_model_v_24 -> in: [batch_size, 1, 14, 32]; out: [batch_size, 1, 14, 32])
        # BatchNorm_CNN2 = model.add(tf.keras.layers.BatchNormalization(name="BATCHNORM_CNN2"))

        ###################################### CNN3 + MaxPool + Activation + BatchNorm #####################################
        ################## conv_model_v_24 -> in: [batch_size, 1, 14, 32]; out: [batch_size, 1, 1, 9] ######################
        # CNN3 params.
        filter_CNN3_height =  filter_CNN3_height 
        if  filter_CNN3_width  == 0:  # if CNN3_width = 0 -> set CNN3_width = width of previous output
            filter_CNN3_width = model.layers[-1].output_shape[2]
        else:
            filter_CNN3_width =  filter_CNN3_width 
        filter_CNN3_chans =  filter_CNN3_chans 
        strides_CNN3_height =  strides_CNN3_height 
        strides_CNN3_width =  strides_CNN3_width 

        filter_CNN3_shape = (filter_CNN3_height, filter_CNN3_width)
        strides_CNN3_shape = (strides_CNN3_height, strides_CNN3_width)

        # CNN3 (conv_model_v_24 -> in: [batch_size, 1, 14, 32]; out: [batch_size, 1, 9, 1])
        CNN3 = model.add(
            tf.keras.layers.Conv2D(
                filter_CNN3_chans,
                filter_CNN3_shape,
                strides=strides_CNN3_shape,
                padding="valid",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    mean= kernel_init_mean ,
                    stddev= kernel_init_stddev ,
                    seed= model_seed ,
                ),
                name="CNN3",
            )
        )

        # ReLU (conv_model_v_24 -> in: [batch_size, 1, 9, 1]; out: [batch_size, 1, 9, 1])
        ReLU_CNN3 = model.add(tf.keras.layers.Activation(activation=tf.keras.activations.relu, name="RELU_CNN3"))

        # BatchNorm (conv_model_v_24 -> in: [batch_size, 1, 9, 1]; out: [batch_size, 1, 9, 1])
        # BatchNorm_CNN3 = model.add(tf.keras.layers.BatchNormalization(name="BATCHNORM_CNN3"))

        ######################################################## FC ########################################################
        ####################### conv_model_v_24 -> in: [batch_size, 1, 1, 32]; out: [batch_size, 1] ########################
        # Flatten (conv_model_v_24 -> in: [batch_size, 1, 9, 1]; out: [batch_size, 1])
        model.add(tf.keras.layers.Flatten(name="CNN3_FLAT"))

        # FC (conv_model_v_24 -> in: [batch_size, 9]; out: [batch_size, 1]|)
        model.add(
            tf.keras.layers.Dense(
                 label_count ,
                # kernel_initializer=tf.keras.initializers.TruncatedNormal(
                #     mean= kernel_init_mean ,
                #     stddev= kernel_init_stddev ,
                #     seed= model_seed ,
                # ),
                activation="sigmoid",
                name="FINAL_FC",
            )
        )

        return model

GENERATOR = KeywordRecognizer


#  ----------------------------------------------------------------------------


RUNNER = TdnnTestRunner


if __name__ == "__main__":
    pytest.main()
