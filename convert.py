import os
import tensorflow as tf
import random
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import config

FLAGS = tf.app.flags.FLAGS

def get_filename_set(data_set):
    labels = []
    filename_set = []

    with open(FLAGS.data_dir + '/label.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list

    for i, label in enumerate(labels):
        list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label)
        for filename in list:
            filename_set.append([i, FLAGS.data_dir  + '/' + data_set + '/' + label + '/' + filename])

    random.shuffle(filename_set)
    return filename_set

def read_jpeg(filename):
    value = tf.read_file(filename)
    decoded_image = tf.image.decode_jpeg(value, channels=FLAGS.depth)
    resized_image = tf.image.resize_images(decoded_image, [FLAGS.raw_height, FLAGS.raw_width])
    resized_image = tf.cast(resized_image, tf.uint8)

    return resized_image

def convert_images(sess, data_set):
    filename_set = get_filename_set(data_set)
    print('filename', filename_set)
    '''
    with open('./data/' + data_set + '_data.bin', 'wb') as f:
        for i in range(0, len(filename_set)):
            resized_image = read_jpeg(filename_set[i][1])

            try:
                image = sess.run(resized_image)
            except Exception as e:
                print e.message
                continue

            #plt.imshow(np.reshape(image.data, [FLAGS.raw_height, FLAGS.raw_width, FLAGS.depth]))
            #plt.show()

            print i, filename_set[i][0], image.shape
            f.write(chr(filename_set[i][0]))
            f.write(image.data)
    '''

def read_raw_images(sess, data_set):
    filename = ['./data/' + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)

    record_bytes = (FLAGS.raw_height) * (FLAGS.raw_width) * FLAGS.depth + 1
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    tf.train.start_queue_runners(sess=sess)

    for i in range(0, 100):
        result = sess.run(record_bytes)
        print i, result[0]
        image = result[1:len(result)]

        #plt.imshow(np.reshape(image, [FLAGS.raw_height, FLAGS.raw_width, FLAGS.depth]))
        #plt.show()

def main(argv = None):
    with tf.Session() as sess:
        convert_images(sess, 'train')
        convert_images(sess, 'eval')
        #read_raw_images(sess, 'eval')

if __name__ == '__main__':
    tf.app.run()
