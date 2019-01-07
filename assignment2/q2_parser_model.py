import argparse
import pickle
import os
import time
import tensorflow as tf

from model import Model
from q2_initialization import xavier_weight_init
from utils.parser_utils import minibatches, load_and_preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_features = 36
    n_classes = 3
    dropout = 0.75  # (p_drop in the handout)
    embed_size = 50
    hidden_size = 400
    batch_size = 512
    n_epochs = 15
    lr = 4e-4
    l2reg = 1e-5
    h2layer = True
    multichannel = False


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.int32, (None, self.config.n_features), name="input")
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.n_classes), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1.0):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        feed_dict = {
            self.input_placeholder: inputs_batch, 
            self.dropout_placeholder: dropout,
        }
        # labels_batch is only available during training, not inference
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates a tf.Variable and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/tf/reshape

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        if self.config.multichannel:
            self.embeddings = tf.Variable(self.pretrained_embeddings)
            e1 = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
            e2 = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_placeholder)
            embeddings = tf.concat([e1, e2], axis=2)
            embeddings = tf.reshape(embeddings, shape=[-1, self.config.n_features*self.config.embed_size*2])
        else:
            self.embeddings = tf.Variable(self.pretrained_embeddings)
            embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
            embeddings = tf.reshape(embeddings, shape=[-1, self.config.n_features*self.config.embed_size])        
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
              Therefore the keep probability should be set to the value of
              (1 - self.dropout_placeholder)

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        ### YOUR CODE HERE
        xavier_initializer = xavier_weight_init()

        self.W1 = tf.Variable(xavier_initializer((int(x.shape[1]), self.config.hidden_size)), name="W1")
        self.b1 = tf.Variable(tf.zeros((self.config.hidden_size)), name="b1")
        h = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)

        if self.config.h2layer:
            self.W2 = tf.Variable(xavier_initializer((self.config.hidden_size, self.config.hidden_size)), name="W2")
            self.b2 = tf.Variable(tf.zeros((self.config.hidden_size)), name="b2")
            h = tf.nn.relu(tf.matmul(h, self.W2) + self.b2)
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)

        self.U = tf.Variable(xavier_initializer((self.config.hidden_size, self.config.n_classes)), name="U")
        self.bu = tf.Variable(tf.zeros((self.config.n_classes)), name="bu")
        pred = tf.nn.relu(tf.matmul(h_drop, self.U) + self.bu)

        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(self.labels_placeholder), 
            logits=pred)
        cross_entropy_loss = tf.reduce_mean(cross_entropies)
        regularization_loss = self.config.l2reg * (tf.nn.l2_loss(self.W1) 
            + tf.nn.l2_loss(self.U)
            )
        if self.config.h2layer:
            regularization_loss += self.config.l2reg * tf.nn.l2_loss(self.W2)

        loss = cross_entropy_loss + regularization_loss
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Use the learning rate from self.config.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        ### END YOUR CODE
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, parser, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches, verbose=1)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])      # keras.utils.Progbar deprecated force argument
        prog.update(n_minibatches)
        
        print("\nEvaluating on dev set", end=' ')
        dev_UAS, _ = parser.parse(dev_set)
        print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print("New best dev UAS! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')
            print()

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()

def getArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodebug", dest="nodebug", action="store_true",
                       help="Disable debug mode.", default=False)
    parser.add_argument("--test", dest="test", action="store_true",
                       help="Run on test set.", default=False)
    return parser.parse_args()


def main(args):
    debug = not args.nodebug
    print("debug: ", debug)
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...", end=' ')
        start = time.time()
        model = ParserModel(config, embeddings)
        parser.model = model
        init_op = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        parser.session = session
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, parser, train_examples, dev_set)

        if args.test and not debug:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, './data/weights/parser.weights')
            print("Final evaluation on test set", end=' ')
            UAS, dependencies = parser.parse(test_set)
            print("- test UAS: {:.2f}".format(UAS * 100.0))
            print("Writing predictions")
            with open('q2_test.predicted.pkl', 'wb') as f:
                pickle.dump(dependencies, f, -1)
            print("Done!")


if __name__ == '__main__':
    main(getArguments())

