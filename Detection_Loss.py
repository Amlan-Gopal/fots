# Imports
import tensorflow as tf

# Reference: https://github.com/jiangxiluning/FOTS.PyTorch
#Loss class for text Detection Branch
class Detection_Loss(tf.keras.losses.Loss):
    
    def __init__(self, from_logits=False,reduction=tf.keras.losses.Reduction.AUTO,name='Loss_layer'):
        super(Detection_Loss, self).__init__(reduction=reduction, name=name)
        
    def call(self, y_true, y_pred):

        #Extract geo_map and score_maps
        y_true_cls=y_true[:,:,:,0]
        y_pred_cls=y_pred[:,:,:,0]
        y_pred_geo=y_pred[:,:,:,1:6]
        y_true_geo=y_true[:,:,:,1:6]
        training_mask=y_true[:,:,:,6]

        #1. Dice Loss
        dice_loss = self.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # we scale classification loss by factor of 0.01 to match the iou loss part
        dice_loss *=0.01

        #2. IOU and Rotation Angle loss
        rbox_loss_ = self.rbox_loss(y_true_cls,y_true_geo,y_pred_geo,training_mask)


        return 100*(rbox_loss_ + dice_loss)
    
    # Dice coefficient loss for score maps
    def dice_coefficient(self, y_true_cls, y_pred_cls,training_mask):
        """Given score maps (y_true and y_pred) and training mask, it computes dice loss"""

        eps = 10**-6
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)

        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    # IOU loss for geomaps
    def rbox_loss(self, y_true_cls,y_true_geo,y_pred_geo,training_mask):
        """Given score and geo maps, it computes IOU loss and rotation angle loss and returns the total loss"""

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        L_AABB = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)

        L_g = L_AABB +  50*L_theta
        L_g=tf.squeeze(L_g,axis=3)

        return tf.reduce_mean(L_g * y_true_cls * training_mask)


