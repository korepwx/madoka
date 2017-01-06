# -*- coding: utf-8 -*-
import os
import shutil
from logging import getLogger

import numpy as np
import six
import tensorflow as tf

from madoka.dataflow import DataFlow
from madoka.train import collect_variable_summaries, SummaryWriter
from madoka.utils import NamedStatusCode
from madoka.utils.tfcompat import variables_initializer
from madoka.utils.tfhelper import (SessionRestorer,
                                   VariableSelector,
                                   get_variable_values,
                                   ensure_default_session,
                                   VariableSetter)
from .base import Trainer

__all__ = ['EnsembleTrainer', 'BaggingTrainer', 'EnsembleTrainingStatus',
           'EnsembleTrainingTermination']


class _ChildModel(object):
    """Ensemble child model.

    Parameters
    ----------
    trainer : Trainer
        The trainer for this model.

    **kwargs
        Other attributes of this model.
    """

    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


class EnsembleTrainingStatus(NamedStatusCode):
    """Status of training a child model in ensemble trainer."""

    # Continue to train next model
    CONTINUE = 0

    # Termination training due to some error
    ERROR = 1

    # Early terminate here
    EARLY_TERMINATION = 2

    # Early terminate here, and discard the latest model
    EARLY_TERMINATION_DROP_LATEST = 3


class EnsembleTrainingTermination(BaseException):
    """Terminate an ensemble trainer with specified code.

    Parameters
    ----------
    status : int
        One status code from `_EnsembleTrainingStatus`.

    message: str
        Extra message of the termination.
    """

    def __init__(self, status, message=None):
        self.status = status
        # we store None instead of empty message
        self.message = message or None

    def __str__(self):
        name = EnsembleTrainingStatus.get_name(self.status)
        if self.message:
            name = name or self.status
            return '%s: %s' % (name, self.message)
        else:
            if not name:
                name = 'EnsembleTrainingTermination(%s)' % (self.status,)
            return name


class EnsembleTrainer(Trainer):
    """Base class for all ensemble trainers.

    An ensemble trainer would combine several other trainers, run these
    trainers one after another, or in a parallel manner.
    Besides, it would settle the checkpoint and summary directories
    for each of these trainers, so that the whole ensemble training process
    can be restored or examined later.

    Most of the arguments specified for this trainer, e.g., `batch_size`,
    will be ignored, except for those within the construction arguments.
    You should set the arguments of child trainers explicitly.

    Parameters
    ----------
    summary_dir : str
        If specified, will write training summaries of each child trainer
        to `summary_dir + "/" + str(child_id)`.

    checkpoint_dir : str
        If specified, will save training checkpoints of each child trainer
        to `checkpoint_dir + "/" + str(child_id)`.

    purge_child_checkpoint : bool
        Whether or not to purge child model's checkpoint directory?

        If True, will clean up the checkpoint directory of a child
        model after it has been trained and its parameters have been
        saved into master's checkpoint directory.

    name : str
        Name of this ensemble trainer.
    """

    # Name of the directory to hold ensemble training checkpoints
    # under the root checkpoint directory specified by `checkpoint_dir`.
    _CKPT_MASTER_DIR_NAME = 'master'

    # Name of the directory to hold ensemble training summaries
    # under the root summary directory specified by `summary_dir`
    _SUMMARY_MASTER_DIR_NAME = _CKPT_MASTER_DIR_NAME

    # Add ensemble training state variables to this collection.
    _ENSEMBLE_TRAINER_VARS = '__ensemble_trainer_variables'

    def __init__(self,
                 summary_dir=None,
                 checkpoint_dir=None,
                 purge_child_checkpoint=True,
                 name='EnsembleTrainer'):
        super(EnsembleTrainer, self).__init__(
            max_steps=1000,     # whatever value it is, it is required
            summary_dir=summary_dir,
            checkpoint_dir=checkpoint_dir,
            name=name
        )
        self.purge_child_checkpoint = purge_child_checkpoint
        self._models = []
        self._trained_model_num = None     # type: int

        # variables to keep the training states
        with self.variable_space():
            self._model_id_var = tf.get_variable(
                'model_id', initializer=0, dtype=tf.int32,
                trainable=False
            )
            self._trained_model_var = tf.get_variable(
                'trained_models', initializer=0, dtype=tf.int32,
                trainable=False
            )
            for v in (self._model_id_var, self._trained_model_var):
                tf.add_to_collection(self._ENSEMBLE_TRAINER_VARS, v)
            self._model_id_setter = VariableSetter(self._model_id_var)
            self._model_id_trained_model_setter = \
                VariableSetter([self._model_id_var, self._trained_model_var])

    @property
    def model_weight_tensor(self):
        """Get the model weight tensor.

        Returns
        -------
        None | tf.Variable | tf.Tensor
            Returns `None` if the trainer does not compute model weights,
            otherwise returns a variable or a tensor.

        Notes
        -----
        The returned tensor stores the trained weights only in current session.
        """
        return None

    @property
    def model_weight(self):
        """Get the model weights of last train.

        Returns
        -------
        None | np.ndarray
            Returns `None` if the trainer does not compute model weights,
            otherwise returns a numpy array.
        """
        return None

    @property
    def trained_model_num(self):
        """Get the number of actually trained models.

        Returns
        -------
        int
        """
        return self._trained_model_num

    def _add(self, trainer, **kwargs):
        """Add a model to this ensemble trainer.

        This method may be called by derived classes, to implement actual
        `add()` method for specifying child models.

        Parameters
        ----------
        trainer : Trainer
            The trainer for this model.

            All necessary arguments of this trainer should be set
            except the data flow, the checkpoint directory and the
            summary directory, which would be set by the ensemble
            trainer.

        **kwargs
            Other attributes of this model.

        Returns
        -------
        self
        """
        if not isinstance(trainer, Trainer):
            raise TypeError('`trainer` is not a Trainer.')
        if isinstance(trainer, EnsembleTrainer):
            raise TypeError('Training an EnsembleTrainer in another '
                            'is not supported.')
        self._models.append(_ChildModel(trainer, **kwargs))
        return self

    def set_data_flow(self, flow, valid_flow=None, shuffle=True):
        if valid_flow is not None:
            raise RuntimeError('Argument `valid_flow` is not supported by '
                               'EnsembleTrainer.')
        return super(EnsembleTrainer, self). \
            set_data_flow(flow, valid_flow, shuffle)

    def set_data(self, *arrays):
        # ensemble trainer does not split validation set.
        return self.set_data_flow(DataFlow.from_numpy(arrays))

    def _prepare_data_for_run(self, flow, valid_flow):
        # This method will only be called in `run()`, which is used to
        # prepare data for `_run()`.  In ensemble trainer, it is usually
        # not necessary to do anything.
        assert(valid_flow is None)
        return flow, valid_flow

    def _build_session_restorer(self,
                                checkpoint_dir,
                                name='EnsembleTrainerRestorer'):
        """Build session restorer for the ensemble trainer."""
        restorable_vars = (
            self.restorable_vars |
            VariableSelector.collection(self._ENSEMBLE_TRAINER_VARS)
        )
        restorable_vars = restorable_vars.select()
        getLogger(__name__).debug(
            '%s: restorable variables: %r',
            self.name, [v.name for v in restorable_vars]
        )
        return SessionRestorer(
            restorable_vars,
            os.path.join(checkpoint_dir, self._CKPT_MASTER_DIR_NAME),
            name=name
        )

    def _build_after_child_summary(self):
        """Build summary operation after a child is trained.

        Returns
        -------
        VariableSelector
        """
        summary_vars = (
            VariableSelector.collection(self._ENSEMBLE_TRAINER_VARS) -
            VariableSelector.list([self._model_id_var, self._trained_model_var])
        )
        summary_vars = summary_vars.select()
        if not summary_vars:
            return None
        return collect_variable_summaries(summary_vars)

    def _prepare_data_flow_for_child(self, model_id, train_flow):
        """Prepare data flow for child model.

        Parameters
        ----------
        model_id : int
            Index of the child model.

        train_flow : DataFlow
            The training data flow.

        Returns
        -------
        DataFlow
        """
        return train_flow

    def _before_training(self, initial_model_id):
        """Notify that we're going to start the training.

        Parameters
        ----------
        initial_model_id : int
            The initial model index to be trained.
        """

    def _after_training(self):
        """Notify that we've already done the training.

        This method will only be called if everything is done successfully.
        If there's any error occurred, or the KeyboardInterrupt is triggered,
        this will not be called.
        """

    def _before_child_training(self, model_id):
        """Notify that we're going to start the training of a child model.

        Parameters
        ----------
        model_id : int
            Index of the child model.
        """

    def _after_child_training(self, model_id):
        """Notify that we've already done the training of a child model.

        Derived classes should override this to push training status
        and error message to EnsembleTrainer.

        Parameters
        ----------
        model_id : int
            Index of the child model.

        Returns
        -------
        (int, str)
            The status code and possibly extra error message of the training.
        """
        return EnsembleTrainingStatus.CONTINUE, ''

    def _reset_training_states(self):
        super(EnsembleTrainer, self)._reset_training_states()
        sess = ensure_default_session()
        reset_vars = tf.get_collection(self._ENSEMBLE_TRAINER_VARS)
        sess.run(variables_initializer(reset_vars))

    def _run(self, checkpoint_dir, train_flow, valid_flow):
        assert(valid_flow is None)

        # collect the variable summaries, and create summary writer
        if self.summary_enabled:
            summary_dir = os.path.realpath(
                os.path.join(self.summary_dir, self._SUMMARY_MASTER_DIR_NAME))
            with self.variable_space():
                self._summary_writer = SummaryWriter(summary_dir)
                summary_op = self._build_after_child_summary()
        else:
            self._summary_writer = None
            summary_op = None

        # build the restorer of maintaining EnsembleTrainer states.
        restorer = self._build_session_restorer(checkpoint_dir)
        restorer.restore()

        # Check the training progress
        # Here `i` is either restored from checkpoint directory,
        # or reset to its initial value by `_reset_training_states()`.
        i, self._trained_model_num = get_variable_values([
            self._model_id_var,
            self._trained_model_var
        ])
        if i > 0:
            getLogger(__name__).info(
                '%s: recovered from checkpoint, training from model %d, '
                'trained %d models.',
                self.name, i, self._trained_model_num
            )
        else:
            getLogger(__name__).info('%s: fresh training started.', self.name)

        # notify that we're starting the training.
        self._before_training(i)

        # run every child trainer
        terminated = False
        model_count = len(self._models)

        while i < model_count and not terminated:
            m = self._models[i]
            # notify that we're going to start the training of a child
            self._before_child_training(i)

            # prepare for the training.
            getLogger(__name__).info(
                '%s: begin to train model %d.', self.name, i)
            df = self._prepare_data_flow_for_child(i, train_flow)

            # copy arguments of the ensemble trainer to child.
            m.trainer.set_data_flow(df, shuffle=self._shuffle_training_flow)
            if self.placeholders is not None:
                m.trainer.set_placeholders(self.placeholders)
            m.trainer.checkpoint_dir = os.path.join(checkpoint_dir, str(i))
            if self.summary_dir:
                m.trainer.summary_dir = os.path.join(self.summary_dir,
                                                     str(i))

            # run the child trainer.
            m.trainer.run()
            getLogger(__name__).info(
                '%s: finished training model %d.', self.name, i)

            # notify that we've already done the training of a child model.
            status, message = self._after_child_training(i)

            # write the master summary
            if summary_op is not None:
                self._summary_writer.write(summary_op, global_step=i)

            # check whether or not early-termination has taken place
            if status == EnsembleTrainingStatus.CONTINUE:
                # increase the trained model counter to keep this model
                self._trained_model_num += 1
            else:
                # deal with possible error
                ex = EnsembleTrainingTermination(status, message)

                # raise immediately if the termination is caused by error
                if status == EnsembleTrainingStatus.ERROR:
                    raise ex

                # keep this model if it is not dropped
                if status != \
                        EnsembleTrainingStatus.EARLY_TERMINATION_DROP_LATEST:
                    self._trained_model_num += 1

                terminated = True
                getLogger(__name__).info(
                    '%s: ensemble training has been early terminated when '
                    'training model %d, due to %s.',
                    self.name, i, ex
                )
                # set the next model id to model_count, so that we will not
                # continue to train after recovery from this checkpoint.
                self._model_id_setter.set(model_count)

            # update the status variables; `i+1` is the next model to train
            self._model_id_trained_model_setter.set([
                i + 1,
                self._trained_model_num
            ])

            # save the master checkpoint
            restorer.save(global_step=i)
            getLogger(__name__).debug(
                '%s: checkpoint saved after training model %d.',
                self.name, i
            )

            # purge the child model's checkpoint directory.
            if self.purge_child_checkpoint and m.trainer.checkpoint_dir:
                try:
                    if os.path.exists(m.trainer.checkpoint_dir):
                        shutil.rmtree(m.trainer.checkpoint_dir)
                except Exception:
                    getLogger(__name__).warn(
                        '%s: failed to purge checkpoint of model %d.',
                        self.name, i, exc_info=1
                    )

            # move to next model
            i += 1

        # notify that we've finished the training.
        self._after_training()
        getLogger(__name__).info('%s: training has stopped, %d child models '
                                 'trained.', self.name, self._trained_model_num)


class BaggingTrainer(EnsembleTrainer):
    """Bagging trainer.

    The bagging trainer trains each underlying model with re-sampled data.
    """

    def _prepare_data_flow_for_child(self, model_id, train_flow):
        return train_flow.resample_once()

    def add(self, trainer):
        """Add a model to this BaggingTrainer.

        Parameters
        ----------
        trainer : Trainer
            The trainer for this model.

            All necessary arguments of this trainer should be set
            except the data flow, the checkpoint directory and the
            summary directory, which would be set by the ensemble
            trainer.

        Returns
        -------
        self
        """
        return self._add(trainer)
