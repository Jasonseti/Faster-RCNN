
""" Basic params """
annot_path = '../../Jupyter Files/Datasets/PascalVOC/Annotations'
image_path = '../../Jupyter Files/Datasets/PascalVOC/JPEGImages'
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
min_side = 256
max_side = 512
stride = 32

""" Anchor Box params """
aspect_ratios = [0.5, 1.0, 2.0]
scales = [2 ** (i / 2) for i in range(0, 8)]

""" Encoder params """
positive_threshold = 0.5
negative_threshold = 0.3
box_variance = [10.0, 10.0, 5.0, 5.0]

""" Model params """
backbone_trainable = False
region_limit = 32
iou_threshold = 0.5
score_threshold = 0.9
roi_size = (3, 3)
