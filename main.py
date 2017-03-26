import tensorflow as tf
from loadMusic.musicXMLio import musicXMLtoArray
from rnnModel.LSTMDoubleLayer import lstm_double

test_arr = musicXMLtoArray('..\\Summer.xml')

lowest_note = 24
highest_note = 102
note_span = highest_note - lowest_note

my_lstm = lstm_double(note_span*2)

x, seq, out = my_lstm.get_dynamicRNN(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outval = out.eval(feed_dict={seq: [len(test_arr[1])],
                                 x: test_arr[1].reshape([len(test_arr[1]), 1, note_span*2])})

print(outval)
