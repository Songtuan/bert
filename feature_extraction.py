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
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               seq_length):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.tokens = tokens
    self.seq_length = seq_length

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


class FeatureExtractor:
    def __init__(self,  FLAGS):
        self.FLAGS = FLAGS
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.FLAGS.vocab_file, do_lower_case=self.FLAGS.do_lower_case)
        bert_config = modeling.BertConfig.from_json_file(self.FLAGS.bert_config_file)
        self.build_graph(bert_config, self.FLAGS.max_seq_length, self.FLAGS.use_tpu)
        self.restore_model(self.FLAGS.init_checkpoint)


    def build_graph(self, bert_config, max_seq_length, use_one_hot_embeddings, is_training=False):
        self.graph = ComputeGraph(bert_config, max_seq_length, use_one_hot_embeddings, is_training)

    def restore_model(self, init_check_point):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, init_check_point)

    def create_feature(self, text, tokenizer, max_seq_length):
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

        seq_length = len(tokens)

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

        return InputFeatures(tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                             segment_ids=segment_ids, seq_length=seq_length)

    def map_tokens_vectors(self, input_features, output_vectors):
        map_token_vector = []
        for i in range(input_features.seq_length):
            token_vector_pair = (input_features.tokens[i], output_vectors[0, i, :].tolist())
            map_token_vector.append(token_vector_pair)
        return map_token_vector

    def get_features(self, text):
        input_features = self.create_feature(text, self.tokenizer, self.FLAGS.max_seq_length)
        outputs = self.sess.run(self.graph.outputs, feed_dict={self.graph.input_ids: input_features.input_ids,
                                                               self.graph.input_mask: input_features.input_mask,
                                                               self.graph.segment_ids: input_features.segment_ids})
        map_token_vector = self.map_tokens_vectors(input_features, outputs)
        return map_token_vector

    def end_session(self):
        self.sess.close()


def get_init_config():
    return FLAGS




