import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from dataset import PascalVOC
from rpn import RegionProposalNetwork
from rcnn import FasterRCNN
import utils


""" PASCAL VOC Dataset """
annotation_files = os.listdir('../../Jupyter Files/Datasets/PascalVOC/Annotations')
train_ds = PascalVOC(annotation_files[:17000], shuffle=True, preprocessing=True)
valid_ds = PascalVOC(annotation_files[17000:], shuffle=True, preprocessing=False)


""" Region Proposal Network """
RPN = RegionProposalNetwork(
    # rpn=tf.keras.models.load_model('Models/rpn'),
)
# print(RPN.rpn.summary())

RPN.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
# RPN.load_optimizer('Models/rpn_optimizer', 2e-5)

# print(RPN.train_step(train_ds.__getitem__(0)))
for epoch in range(1):
    print('Epoch ' + str(epoch + 1) + ':')
    RPN.fit(train_ds, validation_data=valid_ds)
    RPN.save_models()

for _ in range(10):
    utils.visualization.visualize_rpn(train_ds, RPN, region_limit=10)


""" Faster RCNN """
RCNN = FasterRCNN(
    rpn=tf.keras.models.load_model('Models/rpn'),
    # classifier=tf.keras.models.load_model('Models/classifier'),
    # localizer=tf.keras.models.load_model('Models/localizer')
)
# print(RCNN.classifier.summary())
# print(RCNN.localizer.summary())

# RCNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
RCNN.load_optimizer('Models/rcnn_optimizer', 2e-5)

# print(RCNN.train_step(train_ds.__getitem__(0)))
for epoch in range(1):
    print('Epoch ' + str(epoch + 1) + ':')
    RCNN.fit(train_ds, validation_data=valid_ds)
    RCNN.save_models()

for _ in range(10):
    utils.visualization.visualize_rcnn(train_ds, RCNN, region_limit=10)
