import tensorflow as tf
from keras import layers
import pickle
import cv2 as cv

from encoder import LabelEncoder
import params
import utils


class FasterRCNN(tf.keras.Model):
    """ A class to classify and localize for each Region Proposals from RPN
            Input : Image, shape=(h, w, c)
            Output : [a list of class scores, shape=(num_regions, 1),
                      a list of bboxes, shape=(num_regions, 4)]
    """

    def __init__(self, rpn, classifier=None, localizer=None):
        super(FasterRCNN, self).__init__()
        self.encoder = LabelEncoder()
        self._stride = params.stride
        self._roi_size = params.roi_size
        self._region_limit = params.region_limit
        self._iou_threshold = params.iou_threshold
        self._score_threshold = params.score_threshold

        self.backbone = utils.models.get_backbone()
        self.rpn = rpn
        self.classifier = classifier if classifier else self.get_classifier()
        self.localizer = localizer if localizer else self.get_localizer()

        self.class_loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.box_loss_fn = tf.keras.losses.Huber(reduction='none')
        self.class_loss_metric = tf.keras.metrics.Mean(name='class_loss_metric')
        self.bbox_loss_metric = tf.keras.metrics.Mean(name='bbox_loss_metric')

    def get_classifier(self):
        inputs = tf.keras.Input(self._roi_size + (self.backbone.output_shape[-1],))

        x = inputs
        for size in [512, 512]:
            x = layers.ReLU()(x)
            x = layers.Conv2D(size, 3, padding='same')(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(len(params.CLASS_NAMES) + 1, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs, name='Classifier')

    def get_localizer(self):
        inputs = tf.keras.Input(self._roi_size + (self.backbone.output_shape[-1],))

        x = inputs
        for size in [512]:
            x = layers.LeakyReLU(0.1)(x)
            x = layers.Conv2D(size, 3, padding='same')(x)

        x = layers.Flatten()(x)
        for size in [1024]:
            x = layers.LeakyReLU(0.1)(x)
            x = layers.Dense(size)(x)

        outputs = layers.Dense(4)(x)
        return tf.keras.Model(inputs, outputs, name='Localizer')

    def get_region_proposals(self, image_shape, features, region_limit):
        predictions = self.rpn(features)
        pred_scores = tf.reshape(predictions[0], [-1, 1])
        pred_bboxes = tf.reshape(predictions[1], [-1, 4])

        scores = tf.squeeze(pred_scores)
        bboxes = self.encoder.decode_boxes_from_anchors(image_shape, pred_bboxes)

        chosen_indices = tf.image.non_max_suppression(
            bboxes, scores, max_output_size=region_limit,
            iou_threshold=self._iou_threshold, score_threshold=self._score_threshold
        )
        return tf.gather(bboxes, chosen_indices)

    def loss_fn(self, pred_classes, pred_bboxes, class_targets, box_targets, positive_mask):
        class_loss = self.class_loss_fn(class_targets, pred_classes)
        box_loss = self.box_loss_fn(box_targets, pred_bboxes)
        box_loss = tf.where(positive_mask, box_loss, 0.0)

        return tf.reduce_sum(class_loss) / tf.maximum(1.0, tf.reduce_sum(tf.ones_like(positive_mask, tf.float32))), \
               tf.reduce_sum(box_loss) / tf.maximum(1.0, tf.reduce_sum(tf.cast(positive_mask, tf.float32)))


    def train_step(self, training_data):
        image, bboxes, classes = training_data
        features = self.backbone(tf.expand_dims(image, 0))

        region_proposals = self.get_region_proposals(tf.shape(image), features, region_limit=self._region_limit)
        aligned_features = utils.models.roi_align(features, region_proposals / self._stride, self._roi_size)
        class_targets, box_targets, positive_mask = self.encoder.encode_rcnn_labels(
            region_proposals, bboxes, classes
        )

        with tf.GradientTape(persistent=True) as tape:
            pred_classes = self.classifier(aligned_features, training=True)
            pred_bboxes = self.localizer(aligned_features, training=True)

            class_loss, bbox_loss = self.loss_fn(
                pred_classes, pred_bboxes, class_targets, box_targets, positive_mask
            )

        self.optimizer.minimize(class_loss, self.classifier.trainable_variables, tape=tape)
        self.optimizer.minimize(bbox_loss, self.localizer.trainable_variables, tape=tape)

        self.class_loss_metric.update_state(class_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        return {'class': self.class_loss_metric.result(), 'box': self.bbox_loss_metric.result()}

    def test_step(self, testing_data):
        image, bboxes, classes = testing_data
        features = self.backbone(tf.expand_dims(image, 0))

        region_proposals = self.get_region_proposals(tf.shape(image), features, region_limit=self._region_limit)
        aligned_features = utils.models.roi_align(features, region_proposals / self._stride, self._roi_size)
        class_targets, box_targets, positive_mask = self.encoder.encode_rcnn_labels(
            region_proposals, bboxes, classes
        )

        pred_classes = self.classifier(aligned_features, training=False)
        pred_bboxes = self.localizer(aligned_features, training=False)
        class_loss, bbox_loss = self.loss_fn(
            pred_classes, pred_bboxes, class_targets, box_targets, positive_mask
        )

        self.class_loss_metric.update_state(class_loss)
        self.bbox_loss_metric.update_state(bbox_loss)
        return {'class': self.class_loss_metric.result(), 'box': self.bbox_loss_metric.result()}


    def save_models(self):
        self.classifier.save('Models/classifier')
        self.localizer.save('Models/localizer')
        pickle.dump(self.optimizer.get_weights(), open('Models/rcnn_optimizer', 'wb'))

    def load_optimizer(self, optimizer_path, l_rate):
        optimizer = utils.models.load_optimizer(
            optimizer_path, [self.classifier.trainable_variables, self.localizer.trainable_variables], l_rate
        )
        self.compile(optimizer=optimizer)

    def predict_image(self, image, region_limit):
        features = self.backbone(tf.expand_dims(image, 0))
        region_proposals = self.get_region_proposals(tf.shape(image), features, region_limit)
        aligned_features = utils.models.roi_align(features, region_proposals / self._stride, self._roi_size)

        pred_classes = self.classifier(aligned_features)
        pred_bboxes = self.localizer(aligned_features)

        classes = tf.argmax(pred_classes, axis=-1)
        bboxes = self.encoder.decode_boxes_from_regions(region_proposals, pred_bboxes)
        scores = tf.reduce_max(pred_classes, axis=-1)

        # Background Mask
        positive_mask = tf.not_equal(classes, 0)
        classes = tf.boolean_mask(classes - 1, positive_mask)
        bboxes = tf.boolean_mask(bboxes, positive_mask)
        scores = tf.boolean_mask(scores, positive_mask)

        # Non Max Suppression
        if tf.reduce_any(positive_mask):
            chosen_indices = utils.utils.non_max_suppression(bboxes, scores, iou_threshold=self._iou_threshold)
            classes = tf.gather(classes, chosen_indices)
            bboxes = tf.gather(bboxes, chosen_indices)
            scores = tf.gather(scores, chosen_indices)

        return classes, bboxes, scores
