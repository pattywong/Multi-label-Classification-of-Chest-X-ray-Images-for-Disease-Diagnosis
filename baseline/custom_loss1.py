import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def ml_softmax_loss(batch_logits, gt):

    if batch_logits.dtype != gt.dtype:
        gt = tf.cast(gt, batch_logits.dtype)

    # convert input logits to exponential space
    batch_logits = tf.math.exp(batch_logits)
    
    # select only disease logits
    disease_logits = tf.multiply(batch_logits, gt)

    # select only non-disease logits
    non_disease_labels = 1 - gt
    non_disease_logits = tf.multiply(batch_logits, non_disease_labels)
    
    # reduce sum non-disease logits to reduce the dimension
    non_disease_logits = tf.reduce_sum(non_disease_logits, axis=-1)
    
    # for concatenation
    non_disease_logits = tf.multiply(gt, tf.expand_dims(non_disease_logits, axis=1))
    
    # expands dimension disease and non-disease logits by 2 (ex_d, ex_nd)
    ex_d = tf.expand_dims(disease_logits, axis=2)
    ex_nd = tf.expand_dims(non_disease_logits, axis=2)
    ex_gt = tf.expand_dims(gt, axis=2)
    ex_inv_gt = tf.expand_dims(non_disease_labels, axis=2)
    
    # concat disease and non-disease tensors
    concat_tensor = tf.concat([ex_d, ex_nd], axis=2)
    concat_gt = tf.concat([ex_gt, ex_inv_gt], axis=2)
    concat_gt = tf.multiply(concat_gt, ex_gt)
    
    # softmax each pair
    softmax_val = tf.nn.softmax(concat_tensor, axis=-1)
    
    # masks non-disease value out
    mask = tf.multiply(softmax_val, concat_gt)

    # calculate per label loss
    per_label_loss = tf.reduce_sum(mask, axis=-1)

    # calculate per example loss
    per_exm_loss = tf.reduce_sum(per_label_loss, axis=-1)
    
    # normalize by number of assigned label
    num_label = -1 * tf.math.reduce_sum(gt, axis=-1)
    per_exm_loss = tf.math.divide_no_nan(x=per_exm_loss, y=num_label)
    batch_loss = tf.reduce_mean(per_exm_loss)
    
    return per_exm_loss

def cor_loss(batch_logits, gt):
    
    if batch_logits.dtype != gt.dtype:
        gt = tf.cast(gt, batch_logits.dtype)
    
    # pass batch_logits to sigmoid activation function
    logits = tf.math.sigmoid(batch_logits)
    
    # mask only the positive dieseases
    disease_mask = tf.multiply(logits, gt)
    
    # get non-disease labels
    non_disease_labels = 1 - gt
    
    # adding 1 to zero value in logits for the purpose of matrix multiplication
    logits = disease_mask + non_disease_labels
    
    # product all positive classes's probability scores
    per_exm_loss = -1 * tf.math.abs(tf.math.reduce_prod(logits, axis=-1))

    # compute batch average loss
    batch_loss = tf.math.reduce_mean(per_exm_loss)
    
    return batch_loss
    
# Custom loss function = BCE + MSML + CorL
def custom_loss1(y_true, y_pred):
    
    # Create a custom loss function
    alpha = 0.3
    beta = 0.2
    multi_labels_softmax = ml_softmax_loss(y_pred, y_true)
    correlation_loss = cor_loss(y_pred, y_true)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss = bce + (alpha*multi_labels_softmax) + (beta*correlation_loss)
    return loss