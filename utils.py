import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle

from params import CLASS_NAMES, backbone_trainable


class utils:
    @staticmethod
    def convert_to_xywh(boxes):
        return tf.concat([
            (boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]
        ], axis=-1
        )

    @staticmethod
    def convert_to_corners(boxes):
        return tf.concat([
            boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0
        ], axis=-1
        )

    @staticmethod
    def swap_xy(boxes):
        return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)

    @staticmethod
    def normalize_bboxes(image_shape, boxes):
        image_shape = tf.cast(image_shape, tf.float32)
        return tf.stack([
            boxes[..., 0] / image_shape[1],
            boxes[..., 1] / image_shape[0],
            boxes[..., 2] / image_shape[1],
            boxes[..., 3] / image_shape[0]
        ], axis=-1
        )

    @staticmethod
    def compute_iou_matrix(boxes1, boxes2):
        boxes1_corners = utils.convert_to_corners(boxes1)
        boxes2_corners = utils.convert_to_corners(boxes2)
        upper_left = tf.maximum(boxes1_corners[..., None, :2], boxes2_corners[..., :2])
        bottom_right = tf.minimum(boxes1_corners[..., None, 2:], boxes2_corners[..., 2:])
        intersection = tf.maximum(0.0, bottom_right - upper_left)
        intersection_area = intersection[..., 0] * intersection[..., 1]

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        union_area = boxes1_area[..., None] + boxes2_area - intersection_area

        return intersection_area / union_area

    @staticmethod
    def non_max_suppression(bboxes, scores, iou_threshold):
        chosen_indices = []
        indices = tf.ones(tf.shape(bboxes)[0], tf.bool)

        while tf.reduce_all(indices):
            chosen_index = tf.argmax(tf.where(indices, scores, 0.0))
            chosen_boxes = bboxes[chosen_index]
            chosen_indices.append(chosen_index)

            iou_matrix = utils.compute_iou_matrix(chosen_boxes, bboxes)
            iou_mask = tf.less(iou_matrix, iou_threshold)
            indices = tf.logical_and(indices, iou_mask)

        return chosen_indices


class visualization:
    @staticmethod
    def plot_visualization(image, bboxes, classes=None, scores=None, gt_boxes=None):
        plt.style.use('dark_background')
        plt.axis('off')
        plt.imshow(image.numpy().astype('uint8'))
        ax = plt.gca()
        for i, (x, y, w, h) in enumerate(bboxes):
            ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=None, edgecolor='red'))
            text = ''
            if classes is not None:
                text += CLASS_NAMES[int(classes[i])] + ' | ' + str(int(100 * scores[i])) + '%'
            if scores is not None:
                text += ' | ' + str(round(scores[i].numpy(), 2))
            ax.text(x - w / 2, y - h / 2, text, color='white', fontsize=12,
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
        if gt_boxes is not None:
            for x, y, w, h in gt_boxes:
                ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=None, edgecolor='green'))
        plt.show()

    @staticmethod
    def visualize_datasets(datasets, index=None):
        index = index if index is not None else random.randint(0, datasets.__len__())
        image, bboxes, classes = datasets.__getitem__(index)

        visualization.plot_visualization(image, bboxes=bboxes, classes=classes)

    @staticmethod
    def visualize_rpn(dataset, RPN, region_limit=12, index=None):
        index = index if index is not None else random.randint(0, dataset.__len__())
        image, bboxes, classes = dataset.__getitem__(index)
        region_proposals, region_scores = RPN.predict_image(image, region_limit=region_limit)

        visualization.plot_visualization(image, bboxes=region_proposals, scores=region_scores, gt_boxes=bboxes)

    @staticmethod
    def visualize_rcnn(dataset, RCNN, region_limit, index=None):
        index = index if index is not None else random.randint(0, dataset.__len__())
        image, bboxes, classes = dataset.__getitem__(index)
        pred_classes, pred_bboxes, pred_scores = RCNN.predict_image(image, region_limit=region_limit)

        visualization.plot_visualization(image, bboxes=pred_bboxes, classes=pred_classes, scores=pred_scores, gt_boxes=bboxes)


class checks:
    @staticmethod
    def check_encoder(datasets, encoder, index=None):
        index = index if index is not None else random.randint(0, datasets.__len__())
        image, original_bboxes, classes = datasets.__getitem__(index)

        positive_masks, negative_masks, box_targets = encoder.encode_box_targets(tf.shape(image), original_bboxes)
        scores = tf.cast(positive_masks, tf.float32)
        bboxes = encoder.decode_boxes(tf.shape(image), box_targets)

        chosen_indexes = tf.squeeze(tf.where(scores))
        proposed_regions = tf.gather(bboxes, chosen_indexes)

        print(original_bboxes, proposed_regions)


class preprocessing:
    @staticmethod
    def random_flip_horizontal(image, bboxes):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            image_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
            bboxes = tf.stack([
                image_shape[1] - bboxes[:, 0],
                bboxes[:, 1],
                bboxes[:, 2],
                bboxes[:, 3]
            ], axis=-1
            )

        return image, bboxes

    @staticmethod
    def random_resize(image, bboxes, _min_side, _max_side):
        image_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

        # Randomize resize ratio
        min_ratio = _min_side / tf.reduce_min(image_shape)
        max_ratio = _max_side / tf.reduce_max(image_shape)
        ratio = tf.random.uniform([], min_ratio, max_ratio, dtype=tf.float32)

        # Resize image and adjust bounding boxes
        image_shape = tf.cast(image_shape * ratio, tf.int32)
        image = tf.image.resize(image, image_shape)
        bboxes = bboxes * ratio

        return image, bboxes

    @staticmethod
    def pad_image(image, _pad_stride):
        image_shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        padded_image_shape = tf.cast(tf.math.ceil(image_shape / _pad_stride) * _pad_stride, tf.int32)
        image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])
        return image


class models:
    @staticmethod
    def load_optimizer(optimizer_path, list_of_grad_vars, l_rate):
        optimizer_weights = pickle.load(open(optimizer_path, 'rb'))

        optimizer = tf.keras.optimizers.Adam(l_rate)
        for grad_vars in list_of_grad_vars:
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            optimizer.apply_gradients(zip(zero_grads, grad_vars))

        optimizer.set_weights(optimizer_weights)
        return optimizer

    @staticmethod
    def roi_align(features, region_proposals, roi_size):
        # Normalize against feature map
        region_proposals = utils.normalize_bboxes(tf.shape(features)[1:-1] - 1, region_proposals)
        # Convert to (x1, y1, x2, y2)
        region_proposals = utils.convert_to_corners(region_proposals)
        # Swap x and y
        region_proposals = utils.swap_xy(region_proposals)

        aligned_features = tf.image.crop_and_resize(
                features, region_proposals, tf.zeros(tf.shape(region_proposals)[0], tf.int32), roi_size
        )
        return aligned_features

    @staticmethod
    def get_backbone():
        inputs = tf.keras.Input((None, None, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        outputs = tf.keras.applications.ResNet50(include_top=False)(x)

        backbone = tf.keras.Model(inputs, outputs, name='ResNet50_Backbone')
        backbone.trainable = backbone_trainable
        return backbone

    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), alpha, (1.0 - alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1) * 5.0
