# -*- coding: utf-8 -*-

import numpy as np
import six
import tensorflow as tf

from madoka import config
from madoka.train import collect_variable_summaries
from madoka.utils import check_argtype, filter_not_none
from madoka.utils.tfcompat import scalar_summary, merge_summary
from madoka.utils.tfhelper import (get_variable_values,
                                   remove_from_collection,
                                   variable_space, make_function,
                                   batch_collect_predict, Classifier)
from .base import Trainer
from .constraints import EnsembleClassifierBuilder
from .ensemble import (EnsembleTrainer, EnsembleTrainingStatus)

__all__ = ['AdaboostClassifierTrainer']


class AdaboostClassifierTrainer(EnsembleTrainer, EnsembleClassifierBuilder):
    """Adaboost trainer for classifiers.

    The adaboost trainer trains each underlying classifiers with re-weighted,
    data, as well as the weight of each model, according to the classification
    accuracy.

    Thanks a lot to the authors of Scikit-Learn, where we benefit a lot from
    the implementation of `sklearn.ensemble.AdaboostClassifier`.

    Parameters
    ----------
    learning_rate : float
        Learning rate shrinks the contribution of each classifier.

    algorithm: {'SAMME', 'SAMME.R'}
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        If 'SAMME' then use the SAMME discrete boosting algorithm.

        Default is 'SAMME.R', which typically converges faster than SAMME.

    epsilon : float
        Minimum value for a probability to be regarded as zero.

    initial_data_weight : np.ndarray
        Initial data weights.

        If not specified, will set the weight of each example to 1
        (so that the data weights sum up to data size).

    summary_dir : str
        If specified, will write training summaries of each child trainer
        to `summary_dir + "/" + str(child_id)`.

    checkpoint_dir : str
        If specified, will save training checkpoints of each child trainer
        to `checkpoint_dir + "/" + str(child_id)`.

    purge_child_checkpoint : bool
        Whether or not to purge child model's checkpoint directory?

    predict_batch_size : int
        Batch size for evaluating the child classifiers.
        If not specified, will evaluate the child classifiers in one batch.

    name : str
        Name of this ensemble trainer.

    References
    ----------
    [1] Bishop, Christopher M. "Pattern recognition."
        Machine Learning 128 (2006).
    [2] Zhu, Ji, et al. "Multi-class adaboost."
        Ann Arbor 1001.48109 (2006): 1612.
    """

    def __init__(self,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 epsilon=1e-8,
                 initial_data_weight=None,
                 summary_dir=None,
                 checkpoint_dir=None,
                 purge_child_checkpoint=True,
                 predict_batch_size=None,
                 name='AdaboostTrainer'):
        super(AdaboostClassifierTrainer, self).__init__(
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            purge_child_checkpoint=purge_child_checkpoint,
            name=name
        )
        if algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError('Unsupported boosting algorithm %r.' % algorithm)
        if initial_data_weight is not None:
            if not isinstance(initial_data_weight, np.ndarray):
                initial_data_weight = np.asarray(initial_data_weight)
            if isinstance(initial_data_weight.dtype, six.integer_types):
                initial_data_weight = initial_data_weight.astype(config.floatX)

        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.initial_data_weight = initial_data_weight
        self.predict_batch_size = predict_batch_size

        # placeholder for feeding error signals in Adaboost algorithm
        with self.variable_space('adaboost'):
            self._error_ph = tf.placeholder(name='error', shape=(None,),
                                            dtype=tf.bool)

        # lazy initialized members
        self._target_num = None         # type: int
        self._data_weight_var = None    # type: tf.Variable
        self._model_weight_var = None   # type: tf.Variable

        # members to store training results
        self._model_weight = None       # type: np.ndarray

    @property
    def model_weight_tensor(self):
        return self._model_weight_var

    @property
    def model_weight(self):
        return self._model_weight

    def set_placeholders(self, placeholders, label=None):
        ret = super(AdaboostClassifierTrainer, self). \
            set_placeholders(placeholders, label=label)
        label = self.label_placeholder
        if label.get_shape().ndims != 1 or label.dtype != tf.int32:
            raise TypeError('`label` must be 1-D int32 tensor.')
        return ret

    def add(self, trainer, proba, label=None):
        """Add a classifier to this AdaboostTrainer.

        Parameters
        ----------
        trainer : Trainer
            The trainer for this model.

            All necessary arguments of this trainer should be set except
            the data flow, the checkpoint directory and the summary directory,
            which would be set by the ensemble trainer.

        proba : tf.Tensor
            The probability output of the classifier.

            It is required that the 2nd dimension of `proba`, if exist,
            to be determined.  If the `proba` is a 1D tensor, or 2D tensor
            where the 2nd dimension is 1, then it will be regarded as
            the probability of a binary classifier taking positive label.

            This is required for 'SAMME.R' algorithm, but may be `None`
            when using 'SAMME' algorithm, if `label` is specified.

        label : tf.Tensor
            The label output of the classifier.

        Returns
        -------
        self
        """
        if not hasattr(trainer, 'set_weight'):
            raise TypeError('`trainer` is not a WeightedTrainer, thus is not '
                            'supported by AdaboostTrainer.')
        if proba is None:
            if self.algorithm == 'SAMME.R':
                raise TypeError('`proba` is required for SAMME.R algorithm.')
            elif label is None:
                raise TypeError('At least one of `proba` and `label` should '
                                'be specified.')
        else:
            check_argtype(proba, tf.Tensor, 'proba')
            proba_shape = proba.get_shape()
            if proba_shape.ndims == 1 or proba_shape[1].value == 1:
                target_num = 2
            elif proba_shape.ndims != 2 or proba_shape[1].value is None \
                    or proba_shape[1].value == 0:
                raise TypeError('The 2nd dimension of `proba` is required '
                                'to be determined and non-zero.')
            else:
                target_num = proba_shape[1].value

            if not self._models:
                self._target_num = target_num
            elif self._target_num != target_num:
                raise TypeError('The target number of this classifier does not '
                                'match that of others.')

        if label is not None:
            check_argtype(label, tf.Tensor, 'label')
            if label.get_shape().ndims != 1 or label.dtype != tf.int32:
                raise TypeError('`label` should be a 1D int32 tensor.')

        return super(AdaboostClassifierTrainer, self). \
            _add(trainer, proba=proba, label=label)

    def _build_label(self, label, proba):
        if label is None:
            shape = proba.get_shape()
            if shape.ndims == 1 or shape[1].value == 1:
                label = tf.cast(proba >= 0.5, dtype=tf.int32)
                if shape.ndims == 2:
                    label = tf.squeeze(label, [1])
            else:
                label = tf.cast(tf.argmax(proba, 1), dtype=tf.int32)
        elif label.dtype != tf.int32:
            label = tf.cast(label, dtype=tf.int32)
        return label

    def _complete_binary_proba(self, proba):
        shape = proba.get_shape()
        if shape.ndims == 1:
            proba = tf.reshape(proba, [-1, 1])
            shape = proba.get_shape()
        if shape.ndims == 2 and shape[1].value == 1:
            proba = tf.concat(concat_dim=1, values=[1.0 - proba, proba])
        return proba

    def _normalize_data_weight(self, data_weight):
        """Normalize the data weight tensor so that they sum up to data size.

        This method will only do normalization if the data weights sum up to
        positive number.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            The normalized data weight, and a boolean flag to indicate
            whether or not the sum of these weights is above zero.
        """
        data_size = tf.cast(tf.size(data_weight), dtype=data_weight.dtype)
        data_weight_sum = tf.reduce_sum(data_weight)
        positive_flag = data_weight_sum > 0
        normalized = tf.cond(
            positive_flag,
            lambda: data_weight / data_weight_sum * data_size,
            lambda: data_weight
        )
        return normalized, positive_flag

    def _displace_zero_proba(self, proba):
        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        return tf.maximum(
            proba,
            tf.constant(self.epsilon, dtype=proba.dtype)
        )

    def _build_boost_summary(self, incorrect, estimator_error):
        # build summaries
        if self.summary_enabled:
            with variable_space('model_summary'):
                error_summary = scalar_summary(
                    '%s/training_error' % self.scope.name,
                    tf.reduce_mean(tf.cast(incorrect, dtype=tf.float32))
                )
                weighted_error_summary = scalar_summary(
                    '%s/weighted_training_error' % self.scope.name,
                    estimator_error / tf.cast(tf.size(self._data_weight_var),
                                              dtype=estimator_error.dtype)
                )
                data_weight_summary = collect_variable_summaries([
                    self._data_weight_var
                ])
                summary_op = merge_summary([
                    error_summary, weighted_error_summary, data_weight_summary
                ])
        else:
            summary_op = None
        return summary_op

    def _build_model_update(self, model_id, weight):
        return tf.scatter_update(
            self._model_weight_var, [model_id], [weight]
        )

    def _build_boost_real(self, model_id):
        """Prepare for a single boost on specified model using SAMME.R.

        This method builds one or two TensorFlow functions for each model,
        i.e., `m.update_fn` and possibly `m.collect_fn`.  The `m.update_fn`
        is the function which updates data and model weight according to
        SAMME.R algorithm.  The `m.collect_fn`, if it exists, should
        generate the inputs of `update_fn` by mini-batches, in case the
        training flow is too large to be fed into `m.update_fn` in one batch.

        Build Functions
        ---------------
        m.collect_fn : (*args) -> (np.ndarray, np.ndarray)
            Produce probability output with given training data flow,
            along with the training label.

        m.update_fn : (*args) -> (int, str, tf.Summary)
            Update the data and model weight with given training data flow,
            or probability output computed by `m.collect_fn`.

            Returns an integer, one of `EnsembleTrainingStatus`, an extra
            message about the status, as well as the summary of the child
            model's training process.
        """

        # get the tensor of probability output
        m = self._models[model_id]
        y_predict_proba = m.proba
        if y_predict_proba is None:
            raise RuntimeError(
                'SAMME.R requires probability output of the classifier.')

        # build the collect function
        if self.predict_batch_size is None:
            m.collect_fn = None
        else:
            m.collect_fn = make_function(
                inputs=self.placeholders,
                outputs=[y_predict_proba, self.label_placeholder],
                name='collect_fn'
            )
            y_predict_proba = tf.placeholder(
                y_predict_proba.dtype, shape=y_predict_proba.get_shape(),
                name='predict_proba'
            )

        # get the tensor of label prediction
        y_predict = self._build_label(m.label, y_predict_proba)

        # complete the probability tensor if it is binary classifier
        y_predict_proba = self._complete_binary_proba(y_predict_proba)

        # Instance incorrectly classified
        y = self.label_placeholder
        if y.dtype != tf.int32:
            y = tf.cast(y, dtype=tf.int32)
        incorrect = tf.not_equal(y, y_predict)

        # Error fraction
        estimator_error = (
            tf.reduce_sum(tf.boolean_mask(self._data_weight_var, incorrect)) /
            tf.reduce_sum(self._data_weight_var)
        )
        summary_op = self._build_boost_summary(incorrect, estimator_error)

        # Stop if the classification is perfect
        def perfect_branch():
            return tf.tuple(
                filter_not_none([
                    tf.constant(EnsembleTrainingStatus.EARLY_TERMINATION),
                    tf.constant(''),
                    summary_op
                ]),
                control_inputs=[self._build_model_update(model_id, 1.0)]
            )

        # Do boost data if the classification is not perfect
        def update_branch():
            # Construct y coding as described in Zhu et al [2]:
            #
            #    y_k = 1 if c == k else -1 / (K - 1)
            #
            # where K == target_num and c, k in [0, K) are indices along the
            # second axis of the y coding with c being the index corresponding
            # to the true class label.
            target_num = self._target_num
            if y_predict_proba.dtype == tf.float32:
                y_code_dtype = np.float32
            else:
                y_code_dtype = np.float64
            y_codes = np.asarray([-1. / (target_num - 1), 1.],
                                 dtype=y_code_dtype)
            y_coding = tf.gather(
                y_codes,
                tf.one_hot(y, depth=target_num, dtype=tf.int32)
            )

            # Displace zero probabilities
            proba = self._displace_zero_proba(y_predict_proba)

            # Boost weight using multi-class AdaBoost SAMME.R algorithm
            scale = tf.constant(
                -1. * self.learning_rate * ((target_num - 1.) / target_num),
                dtype=proba.dtype
            )
            inner1d = tf.reduce_sum(
                tf.mul(y_coding, tf.log(y_predict_proba)),
                [1]
            )
            estimator_weight = scale * inner1d

            # only boost positive weights
            data_weight = self._data_weight_var
            data_weight_mask = tf.logical_or(data_weight > 0,
                                             estimator_weight < 0)
            new_data_weight = (
                data_weight * tf.exp(
                    tf.select(
                        data_weight_mask,
                        estimator_weight,
                        tf.zeros(tf.shape(estimator_weight),
                                 dtype=estimator_weight.dtype)
                    )
                )
            )
            new_data_weight, is_data_weight_positive = \
                self._normalize_data_weight(new_data_weight)

            data_weight_update = \
                tf.assign(self._data_weight_var, new_data_weight)
            return tf.tuple(
                filter_not_none([
                    tf.cond(
                        is_data_weight_positive,
                        lambda: tf.constant(
                            EnsembleTrainingStatus.CONTINUE,
                            dtype=tf.int32
                        ),
                        lambda: tf.constant(
                            EnsembleTrainingStatus.EARLY_TERMINATION,
                            dtype=tf.int32
                        )
                    ),
                    tf.constant(''),
                    summary_op
                ]),
                control_inputs=[data_weight_update,
                                self._build_model_update(model_id, 1.0)]
            )

        # Build the update function
        out = tf.cond(
            estimator_error > 0,
            update_branch,
            perfect_branch
        )
        if self.predict_batch_size is None:
            update_ph = self.placeholders
        else:
            update_ph = (y_predict_proba, self.label_placeholder)
        m.update_fn = make_function(inputs=update_ph, outputs=out,
                                    name='update_fn')

    def _build_boost_discrete(self, model_id):
        """Prepare for a single boost on specified model using SAMME

        This methods builds one or two TensorFlow functions for each model,
        i.e., `m.update_fn` and possibly `m.collect_fn`, just as the method
        `_build_boost_real()` does.
        """
        # get the tensor of label output
        m = self._models[model_id]
        y_predict = self._build_label(m.label, m.proba)

        # build the collect function
        if self.predict_batch_size is None:
            m.collect_fn = None
        else:
            m.collect_fn = make_function(
                inputs=self.placeholders,
                outputs=[y_predict, self.label_placeholder],
                name='collect_fn'
            )
            y_predict = tf.placeholder(tf.int32, shape=(None,), name='predict')

        # Instances incorrectly classified
        y = self.label_placeholder
        if y.dtype != tf.int32:
            y = tf.cast(y, dtype=tf.int32)
        incorrect = tf.not_equal(y, y_predict)

        # Error fraction
        estimator_error = (
            tf.reduce_sum(tf.boolean_mask(self._data_weight_var, incorrect)) /
            tf.reduce_sum(self._data_weight_var)
        )
        summary_op = self._build_boost_summary(incorrect, estimator_error)

        # Stop if the classification is perfect
        def perfect_branch():
            return tf.tuple(
                filter_not_none([
                    tf.constant(EnsembleTrainingStatus.EARLY_TERMINATION),
                    tf.constant(''),
                    summary_op
                ]),
                control_inputs=[self._build_model_update(model_id, 1.0)]
            )

        # Stop if the error is at least as bad as random guessing
        def random_guess_branch():
            if model_id == 0:
                ret = tf.tuple(
                    filter_not_none([
                        tf.constant(EnsembleTrainingStatus.ERROR),
                        tf.constant(
                            'Base classifier trained by Adaboost algorithm '
                            'is worse than random, ensemble cannot be fit.'
                        ),
                        summary_op
                    ])
                )
            else:
                status = EnsembleTrainingStatus.EARLY_TERMINATION_DROP_LATEST
                ret = tf.tuple(
                    filter_not_none([
                        tf.constant(status),
                        tf.constant(''),
                        summary_op
                    ])
                )
            return ret

        # Do boost data if the classification is not perfect
        def update_branch():
            # Boost weight using multi-class AdaBoost SAMME alg
            scale = tf.constant(
                self.learning_rate,
                dtype=estimator_error.dtype
            )
            bias = tf.constant(
                self.learning_rate * np.log(self._target_num - 1.),
                dtype=estimator_error.dtype
            )
            estimator_weight = bias + scale * tf.log(
                (1. - estimator_error) / estimator_error)

            # only boost positive weights
            data_weight = self._data_weight_var
            data_weight_mask = tf.logical_and(
                incorrect,
                tf.logical_or(data_weight > 0, estimator_weight < 0)
            )
            new_data_weight = (
                data_weight * tf.exp(
                    estimator_weight *
                    tf.cast(data_weight_mask, dtype=tf.float32)
                )
            )
            new_data_weight, is_data_weight_positive = \
                self._normalize_data_weight(new_data_weight)

            data_weight_update = \
                tf.assign(self._data_weight_var, new_data_weight)
            return tf.tuple(
                filter_not_none([
                    tf.cond(
                        is_data_weight_positive,
                        lambda: tf.constant(
                            EnsembleTrainingStatus.CONTINUE,
                            dtype=tf.int32
                        ),
                        lambda: tf.constant(
                            EnsembleTrainingStatus.EARLY_TERMINATION,
                            dtype=tf.int32
                        )
                    ),
                    tf.constant(''),
                    summary_op
                ]),
                control_inputs=[
                    data_weight_update,
                    self._build_model_update(model_id, estimator_weight)
                ]
            )

        # Build the update function
        err_threshold = tf.constant(1. - (1. / self._target_num),
                                    dtype=estimator_error.dtype)
        out = tf.cond(
            estimator_error >= err_threshold,
            random_guess_branch,
            lambda: tf.cond(
                estimator_error > 0,
                update_branch,
                perfect_branch
            )
        )
        if self.predict_batch_size is None:
            update_ph = self.placeholders
        else:
            update_ph = (y_predict, self.label_placeholder)
        m.update_fn = make_function(inputs=update_ph, outputs=out,
                                    name='update_fn')

    def ensemble_classifier(self, input_ph):
        """Build a classifier upon the ensemble models.

        This method is only guaranteed to work after the ensemble models
        have already been trained.

        Parameters
        ----------
        input_ph : tf.Tensor | collections.Iterable[tf.Tensor]
            The input placeholders, except label.

        Returns
        -------
        Classifier
        """
        if self._model_weight is None:
            raise RuntimeError('Adaboost classifier has not been trained.')

        def real_decision(m):
            proba = m.proba
            if proba is None:
                raise RuntimeError('`proba` tensor is required for SAMME.R')
            # complete the probability tensor if it is binary classifier
            proba = self._complete_binary_proba(proba)

            # compose the pseudo-probability according to SAMME.R
            scale1 = tf.constant(self._target_num - 1, dtype=proba.dtype)
            scale2 = tf.constant(1. / self._target_num, dtype=proba.dtype)
            proba = self._displace_zero_proba(proba)
            log_proba = tf.log(proba)
            log_proba_sum = tf.reduce_sum(log_proba, [1], keep_dims=True)
            return scale1 * (log_proba - scale2 * log_proba_sum)

        def discrete_decision(m):
            label = self._build_label(m.label, m.proba)
            dtype = self._model_weight_var.dtype.base_dtype
            return tf.one_hot(label, depth=self._target_num, dtype=dtype)

        with self.variable_space('ensemble_classifier'):
            model_count = self._trained_model_num
            models = self._models[: model_count]

            # build the pseudo-probability tensor
            with variable_space('pseudo_proba'):
                model_weight = self._model_weight[: model_count]
                w = model_weight.reshape([model_count, 1, 1])
                total_weight = tf.constant(np.sum(model_weight))

                # compute the pseudo-probability
                if self.algorithm == 'SAMME.R':
                    pseudo_proba = tf.pack(
                        [real_decision(m) for m in models])
                else:
                    pseudo_proba = tf.pack(
                        [discrete_decision(m) for m in models])

                assert(pseudo_proba.get_shape().ndims == 3)
                pseudo_proba = tf.reduce_sum(w * pseudo_proba, [0])
                assert(pseudo_proba.get_shape().ndims == 2)

                if total_weight.dtype != pseudo_proba.dtype:
                    total_weight = tf.cast(total_weight,
                                           dtype=pseudo_proba.dtype)
                pseudo_proba = pseudo_proba / total_weight
                scale = tf.constant(1. / (self._target_num - 1),
                                    dtype=pseudo_proba.dtype)
                pseudo_proba = scale * pseudo_proba

            # build the label prediction
            with variable_space('label'):
                label = tf.argmax(pseudo_proba, 1)

            # build the probability output (normalize the pseudo-probability)
            with variable_space('proba'):
                unnormalized_proba = tf.exp(pseudo_proba)
                proba_normalizer = tf.reduce_sum(unnormalized_proba, [1],
                                                 keep_dims=True)
                proba = unnormalized_proba / proba_normalizer

            # build the log probability output
            with variable_space('log_proba'):
                log_proba = pseudo_proba - tf.log(proba_normalizer)

            return Classifier(input_ph, proba, label=label, log_proba=log_proba)

    def _prepare_variables_for_run(self, train_flow, valid_flow):
        data_size = len(train_flow)
        model_num = len(self._models)

        # get the initial data weight and model weight
        if self.initial_data_weight is None:
            data_weight = np.ones(shape=[data_size], dtype=config.floatX)
        elif data_size != len(self.initial_data_weight):
            raise ValueError('Length of data != length of initial data weight.')
        else:
            data_weight = self.initial_data_weight
        model_weight = np.zeros(shape=[model_num], dtype=config.floatX)

        # Purge old variables from `training_states` collection
        #
        # These variables are created during last train, and may have been
        # used to construct a predictor.  Thus we should not reset them in
        # ensemble training.
        for v in (self._data_weight_var, self._model_weight_var):
            if v is not None:
                remove_from_collection(self._ENSEMBLE_TRAINER_VARS, v)

        # now create the new variables
        with self.variable_space():
            self._data_weight_var = tf.get_variable(
                'data_weight', initializer=data_weight, trainable=False)
            self._model_weight_var = tf.get_variable(
                'model_weight', initializer=model_weight, trainable=False)
            for v in (self._data_weight_var, self._model_weight_var):
                tf.add_to_collection(self._ENSEMBLE_TRAINER_VARS, v)

        # build the computation graph of Adaboost algorithm for each child
        with self.variable_space('adaboost'):
            for i in range(model_num):
                with variable_space('child%d' % i):
                    if self.algorithm == 'SAMME.R':
                        self._build_boost_real(i)
                    else:
                        self._build_boost_discrete(i)

        self._model_weight = None
        return super(AdaboostClassifierTrainer, self). \
            _prepare_variables_for_run(train_flow, valid_flow)

    def _build_after_child_summary(self):
        return scalar_summary(
            '%s/model_weight' % self.scope.name,
            tf.gather(self._model_weight_var, self._model_id_var)
        )

    def _before_child_training(self, model_id):
        m = self._models[model_id]
        m.trainer.set_weight(self._data_weight_var)
        return super(AdaboostClassifierTrainer, self). \
            _before_child_training(model_id)

    def _after_child_training(self, model_id):
        m = self._models[model_id]
        flow = self._flow
        if m.collect_fn is None:
            # error_fn is None, indicating we should feed the whole data flow
            # into the function in one batch.
            args = flow.all()
        else:
            # otherwise we should collect the error signal first
            args = batch_collect_predict(
                m.collect_fn, flow, self.predict_batch_size)

        # call the update function
        results = m.update_fn(*args)
        if len(results) == 2:
            status, message = results
            summary = None
        else:
            status, message, summary = results

        # write the summary
        if summary is not None:
            self._summary_writer.write(summary, global_step=model_id)

        return status, message

    def _after_training(self):
        self._model_weight = get_variable_values(self._model_weight_var)
        return super(AdaboostClassifierTrainer, self)._after_training()

