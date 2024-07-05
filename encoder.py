import tensorflow as tf
import utils
import params


class AnchorBox:
    """ A class to assign anchors of various sizes and shapes across all feature maps levels
            Anchor box dims   : (5, height/stride * width/stride * anchor dimensions, 4)
            Anchor box formats: (center_x, center_y, width, height)
    """

    def __init__(self):
        self.aspect_ratios = params.aspect_ratios
        self.scales = params.scales
        self.stride = params.stride
        self.area = self.stride ** 2

        self.num_anchors = len(self.aspect_ratios) * len(self.scales)
        self.anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(self.area / ratio)
                anchor_width = self.area / anchor_height
                dims = tf.stack([anchor_width, anchor_height], axis=-1)
                anchor_dims.append(scale * dims)
        return tf.stack(anchor_dims, axis=0)

    def get_anchors(self, image_shape):
        feature_height = image_shape[0] / self.stride
        feature_width = image_shape[1] / self.stride

        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self.stride

        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self.num_anchors, 1])

        dims = tf.reshape(self.anchor_dims, [1, 1, self.num_anchors, 2])
        dims = tf.tile(dims, [feature_height, feature_width, 1, 1])

        anchor_boxes = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchor_boxes, [-1, 4])



class LabelEncoder:
    """ A class to encode target matrices from lists of labels """

    def __init__(self):
        self.AnchorGenerator = AnchorBox()

        self._positive_threshold = params.positive_threshold
        self._negative_threshold = params.negative_threshold
        self._box_variance = tf.cast(params.box_variance, tf.float32)


    def encode_box_targets(self, image_shape, gt_boxes):
        """ Encode bboxes and classes as object masks and box targets
                Input : [image, shape=(h, w, c),
                         bboxes, shape=(num_obj, 4), (center_x, center_y, width, height)]
                Output: [box shift targets, shape=(levels, h/stride * w/stride * num_anchors, 4),
                         positive object mask, shape=(h/stride * w/stride * num_anchors),
                         negative object mask, shape=(h/stride * w/stride * num_anchors)]
        """

        anchor_boxes = self.AnchorGenerator.get_anchors(image_shape)
        iou_matrix = utils.utils.compute_iou_matrix(anchor_boxes, gt_boxes)

        # Object Masks
        max_iou = tf.reduce_max(iou_matrix, axis=-1)
        positive_masks = tf.greater_equal(max_iou, self._positive_threshold)
        negative_masks = tf.less(max_iou, self._negative_threshold)

        # Make sure every gt_box is included
        indices = tf.cast(tf.argmax(iou_matrix, axis=0), tf.int32)
        positive_masks = tf.tensor_scatter_nd_update(
            tf.cast(positive_masks, tf.float32), tf.expand_dims(indices, -1), tf.ones(tf.shape(indices))
        )
        positive_masks = tf.cast(tf.minimum(positive_masks, 1.0), tf.bool)

        # Box Targets
        matched_boxes = tf.gather(gt_boxes, tf.argmax(iou_matrix, axis=-1))
        box_targets = tf.concat([
                (matched_boxes[..., :2] - anchor_boxes[..., :2]) / anchor_boxes[..., 2:],
                tf.math.log(matched_boxes[..., 2:] / anchor_boxes[..., 2:])
            ], axis=-1
        )
        box_targets = box_targets * self._box_variance

        return positive_masks, negative_masks, box_targets


    def encode_rcnn_labels(self, region_proposals, gt_boxes, classes):
        """ Encode bboxes and classes as box targets and sparse categorical array
                Input : [region proposals, shape=(n, 4), (center_x, center_y, width, height)
                         bboxes, shape=(n, 4), (center_x, center_y, width, height),
                         classes, shape=(n,)]
                Output: [class targets, shape=(n + 1,), range(0, num_classes + 1), 0 equals background,
                         box shift targets, shape=(h/stride * w/stride * num_anchors, 4),
                         positive object mask, shape=(h/stride * w/stride * num_anchors)]
        """

        iou_matrix = utils.utils.compute_iou_matrix(region_proposals, gt_boxes)
        positive_masks = tf.greater_equal(tf.reduce_max(iou_matrix, axis=-1), self._positive_threshold)

        # Class Targets
        matched_classes = tf.gather(classes, tf.argmax(iou_matrix, axis=-1))
        matched_classes = tf.where(positive_masks, matched_classes + 1, 0)
        class_targets = tf.one_hot(matched_classes, len(params.CLASS_NAMES) + 1)

        # Box Targets
        matched_boxes = tf.gather(gt_boxes, tf.argmax(iou_matrix, axis=-1))
        box_targets = tf.concat([
                (matched_boxes[..., :2] - region_proposals[..., :2]) / region_proposals[..., 2:],
                tf.math.log(matched_boxes[..., 2:] / region_proposals[..., 2:])
            ], axis=-1
        )
        box_targets = box_targets * self._box_variance

        return class_targets, box_targets, positive_masks


    def decode_boxes_from_anchors(self, image_shape, box_targets):
        anchor_boxes = self.AnchorGenerator.get_anchors(image_shape)
        box_targets = box_targets / self._box_variance
        bboxes = tf.concat([
                box_targets[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
                tf.math.exp(box_targets[..., 2:]) * anchor_boxes[..., 2:]
            ], axis=-1
        )
        return bboxes

    def decode_boxes_from_regions(self, region_proposals, box_targets):
        box_targets = box_targets / self._box_variance
        bboxes = tf.concat([
                box_targets[..., :2] * region_proposals[..., 2:] + region_proposals[..., :2],
                tf.math.exp(box_targets[..., 2:]) * region_proposals[..., 2:]
            ], axis=-1
        )
        return bboxes
