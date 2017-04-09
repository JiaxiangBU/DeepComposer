import tensorflow as tf
from loadMusic.musicXMLio import musicXMLtoArray
from rnnModel.LSTMDoubleLayer import lstm_double

test_arr = musicXMLtoArray('..\\Summer.xml')
print(test_arr[1].shape)

lowest_note = 24
highest_note = 102
note_span = highest_note - lowest_note

my_lstm = lstm_double(note_span*2)
#print(my_lstm.first_cell)
#print(my_lstm.second_cell)

my_lstm.get_dynamicRNN(1)
my_lstm.setTrainer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        my_lstm.optim.run(feed_dict={my_lstm.seq_length: [len(test_arr[1])-1],
                                     my_lstm.batch_x: test_arr[1][:-1].reshape([len(test_arr[1])-1, 1, note_span * 2]),
                                     my_lstm.batch_y: test_arr[1][1:].reshape([len(test_arr[1])-1, 1, note_span * 2])})
        if i%5 == 0:
            loss = my_lstm.loss.eval(feed_dict={my_lstm.seq_length: [len(test_arr[1])-1],
                                                my_lstm.batch_x: test_arr[1][:-1].reshape([len(test_arr[1])-1, 1, note_span * 2]),
                                                my_lstm.batch_y: test_arr[1][1:].reshape([len(test_arr[1])-1, 1, note_span * 2])})
            print(loss)

#print(outval)
