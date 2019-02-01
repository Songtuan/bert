import random
import modeling
import optimization
import numpy as np
import tokenization
import tensorflow as tf


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", 'Model/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "init_checkpoint", 'Model/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string("vocab_file", 'Model/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids

class ComputeGraph:

    def __init__(self,bert_config, max_seq_length, use_one_hot_embeddings, is_training=False):
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length])
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length])
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None,max_seq_length])

        self.model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=self.input_ids,
        input_mask=self.input_mask,
        token_type_ids=self.segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

        self.outputs = self.model.get_sequence_output()


def create_feature(text, tokenizer, max_seq_length):
    tokens_a = tokenizer.tokenize(text)
    tokens_b = None

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    input_ids = np.asarray(input_ids).reshape(1, max_seq_length)
    input_mask = np.asarray(input_mask).reshape(1, max_seq_length)
    segment_ids = np.asarray(segment_ids).reshape(1, max_seq_length)

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)

def build_graph(bert_config, max_seq_length, use_one_hot_embeddings, is_training=False):
    graph = ComputeGraph(bert_config, max_seq_length, use_one_hot_embeddings, is_training)
    return graph

if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    graph = build_graph(bert_config, FLAGS.max_seq_length, FLAGS.use_tpu)
    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, FLAGS.init_checkpoint)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                print("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        while True:
            input_text = input("Input the text:")
            if input_text == 'q':
                break
            input_features = create_feature(input_text, tokenizer, FLAGS.max_seq_length)
            outputs = graph.outputs.eval(feed_dict={graph.input_ids: input_features.input_ids,
                                                       graph.input_mask: input_features.input_mask,
                                                       graph.segment_ids: input_features.segment_ids})
            print(outputs)




