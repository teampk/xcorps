import tensorflow as tf
import numpy as np
import model
import datetime
import input

FLAGS = tf.app.flags.FLAGS

def eval():
    keep_prob = tf.placeholder(tf.float32)
    images, labels = input.get_data('eval', FLAGS.batch_size)
    hypothesis, cross_entropy, train_step = model.make_network(images, labels, keep_prob)
    print ('Start Evaluation......')
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        max_steps = 10
        true_count = 0.
        total_sample_count = max_steps * FLAGS.batch_size

        top_k_op = tf.nn.in_top_k(hypothesis, labels, 1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(0, max_steps):
            start = datetime.datetime.now()
            predictions = sess.run(top_k_op, feed_dict={keep_prob: 1.0})
            #print('prediction:', predictions)
            true_count += np.sum(predictions)
            print ('Step : %d' %i)
        print ('\nFinish Evaluation......')
        coord.request_stop()
        coord.join(threads)

    print ('precision : %f' % (true_count / total_sample_count))

def main(argv = None):
    eval()

if __name__ == '__main__':
    tf.app.run()
