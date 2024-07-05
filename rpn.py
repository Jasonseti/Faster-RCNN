import tensorflow as tf
from keras import layers
import pickle

from encoder import LabelEncoder
import utils
import params


class RegionProposalNetwork(tf.keras.Model):
    """ A model to predict which regions has an object in it.
            Input : Image, shape=(h, w, c)
            Output : [a matrix of object scores, shape=(levels, h/stride * w/stride * num_anchors),
                      a matrix of bboxes shifts, shape=(levels, h/stride * w/stride * num_anchors, 4)
    """

    def __init__(self, rpn=None):
        super(RegionProposalNetwork, self).__init__()
        self.encoder = LabelEncoder()
        self.num_anchors = self.encoder.AnchorGenerator.num_anchors
        self._iou_threshold = params.iou_threshold
        self._score_threshold = params.score_threshold

        self.backbone = utils.models.get_backbone()
        self.rpn = rpn if rpn else self.get_rpn()

        self.obj_loss_fn = tf.keras.losses.BinaryCrossentropy(reduction='none')
        self.box_loss_fn = tf.keras.losses.Huber(reduction='none')
        self.pos_loss_metric = tf.keras.metrics.Mean(name='pos_loss_metric')
        self.neg_loss_metric = tf.keras.metrics.Mean(name='neg_loss_metric')
        self.box_loss_metric = tf.keras.metrics.Mean(name='box_loss_metric')

    def get_rpn(self):
        inputs = tf.keras.Input((None, None, self.backbone.output_shape[-1]))

        x = inputs
        for size in [512, 512]:
            x = layers.ReLU()(x)
            x = layers.Conv2D(size, 3, padding='same')(x)

        obj_scores = layers.Conv2D(1 * self.num_anchors, 1, activation='sigmoid')(layers.Dropout(0.2)(x))
        obj_bboxes = layers.Conv2D(4 * self.num_anchors, 1)(x)

        return tf.keras.Model(inputs, [obj_scores, obj_bboxes], name='RegionProposalNetwork')

    def get_predictions(self, x, training=False):
        x = tf.expand_dims(x, 0)
        features = self.backbone(x, training=training)
        predictions = self.rpn(features, training=training)
        pred_scores = tf.reshape(predictions[0], [-1, 1])
        pred_bboxes = tf.reshape(predictions[1], [-1, 4])
        return pred_scores, pred_bboxes

    def loss_fn(self, pred_scores, pred_boxes, positive_mask, negative_mask, true_boxes):
        positive_loss = self.obj_loss_fn(tf.ones_like(pred_scores), pred_scores)
        positive_loss = tf.where(positive_mask, positive_loss, 0.0)

        negative_loss = self.obj_loss_fn(tf.zeros_like(pred_scores), pred_scores)
        negative_loss = tf.where(negative_mask, negative_loss, 0.0)

        box_loss = self.box_loss_fn(true_boxes, pred_boxes)
        box_loss = tf.where(positive_mask, box_loss, 0.0)

        return tf.reduce_sum(positive_loss) / tf.maximum(1.0, tf.reduce_sum(tf.cast(positive_mask, tf.float32))), \
               tf.reduce_sum(negative_loss) / tf.maximum(1.0, tf.reduce_sum(tf.cast(negative_mask, tf.float32))), \
               tf.reduce_sum(box_loss) / tf.maximum(1.0, tf.reduce_sum(tf.cast(positive_mask, tf.float32)))


    def train_step(self, training_data):
        image, bboxes, classes = training_data
        positive_masks, negative_masks, box_targets = self.encoder.encode_box_targets(tf.shape(image), bboxes)

        with tf.GradientTape() as tape:
            pred_scores, pred_bboxes = self.get_predictions(image, training=True)

            positive_loss, negative_loss, bbox_loss = self.loss_fn(
                pred_scores, pred_bboxes,
                positive_masks, negative_masks, box_targets
            )
            total_loss = positive_loss + negative_loss + bbox_loss

        self.optimizer.minimize(total_loss, self.rpn.trainable_variables, tape=tape)

        self.pos_loss_metric.update_state(positive_loss)
        self.neg_loss_metric.update_state(negative_loss)
        self.box_loss_metric.update_state(bbox_loss)
        return {'object': self.pos_loss_metric.result(),
                'no object': self.neg_loss_metric.result(),
                'box': self.box_loss_metric.result()}

    def test_step(self, testing_data):
        image, bboxes, classes = testing_data
        positive_masks, negative_masks, box_targets = self.encoder.encode_box_targets(tf.shape(image), bboxes)

        pred_scores, pred_bboxes = self.get_predictions(image, training=False)
        positive_loss, negative_loss, bbox_loss = self.loss_fn(
            pred_scores, pred_bboxes,
            positive_masks, negative_masks, box_targets
        )

        self.pos_loss_metric.update_state(positive_loss)
        self.neg_loss_metric.update_state(negative_loss)
        self.box_loss_metric.update_state(bbox_loss)
        return {'object': self.pos_loss_metric.result(),
                'no object': self.neg_loss_metric.result(),
                'box': self.box_loss_metric.result()}


    def save_models(self):
        self.rpn.save('Models/rpn')
        pickle.dump(self.optimizer.get_weights(), open('Models/rpn_optimizer', 'wb'))

    def load_optimizer(self, optimizer_path, l_rate):
        optimizer = utils.models.load_optimizer(
            optimizer_path, [self.rpn.trainable_variables], l_rate
        )
        self.compile(optimizer=optimizer)

    def predict_image(self, image, region_limit):
        pred_scores, pred_bboxes = self.get_predictions(image)

        scores = tf.squeeze(pred_scores)
        bboxes = self.encoder.decode_boxes_from_anchors(tf.shape(image), pred_bboxes)

        chosen_indices = tf.image.non_max_suppression(
            bboxes, scores, max_output_size=region_limit,
            iou_threshold=self._iou_threshold, score_threshold=self._score_threshold
        )
        return tf.gather(bboxes, chosen_indices), tf.gather(scores, chosen_indices)
