'''
TensorFlow Implementation of "Speaker-Independent Speech Separation with Deep Attractor Network"

TODO docs
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from math import sqrt, isnan
from random import randint
import argparse
from sys import stdout
from collections import OrderedDict
from functools import reduce
from colorsys import hsv_to_rgb
import sys
import os
import copy
import datetime as datetime


import numpy as np
import tensorflow as tf

from app.hparams import hparams
import app.ops as ops


def _dict_add(dst, src):
    for k,v in src.items():
        if k not in dst:
            dst[k] = v
        else:
            dst[k] += v


def _dict_mul(di, coeff):
    for k,v in di.items():
        di[k] = v * coeff


def _dict_format(di):
    return ' '.join('='.join((k, str(v))) for k,v in di.items())


class Model(object):
    '''
    Base class for a fully trainable model  (完全可训练模型的基本类)

    Should be singleton(需要时是单例的)
    '''
    def __init__(self, name='BaseModel'):
        self.name = name
        self.s_states_di = {}
        self.v_learn_rate = tf.Variable(
            hparams.LR,
            trainable=False,
            dtype=hparams.FLOATX,
            name='learn_rate')

    def lyr_lstm(
            self, name, s_x, hdim,
            axis=-1, t_axis=0,
            op_linear=ops.lyr_linear,
            w_init=None, b_init=None):
        '''
        这个函数应该是用来生成lstm的
        Args:
            name: string
            s_x: input tensor(输入张量)
            hdim: size of hidden layer(隐藏层的大小)
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_lstm
        '''
        x_shp = s_x.get_shape().as_list()
        ndim = len(x_shp)
        assert -ndim <= axis < ndim
        assert -ndim <= t_axis < ndim
        axis = axis % ndim
        t_axis = t_axis % ndim
        assert axis != t_axis
        # make sure t_axis is 0, to make scan work
        perm = []
        if t_axis != 0:
            if axis == 0:
                axis = t_axis % ndim
            perm = list(range(ndim))
            perm[0], perm[t_axis] = perm[t_axis], perm[0]
            s_x = tf.transpose(s_x, perm)
        x_shp[t_axis], x_shp[0] = x_shp[0], x_shp[t_axis]
        idim = x_shp[axis]
        assert isinstance(idim, int)
        h_shp = copy.copy(x_shp[1:])
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            v_hid = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='hid',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell
            self.s_states_di[v_hid.name] = v_hid

            op_lstm = lambda _h, _x: ops.lyr_lstm_flat(
                name='LSTM',
                s_x=_x, v_cell=_h[0], v_hid=_h[1],
                axis=axis-1, op_linear=op_linear,
                w_init=w_init, b_init=b_init)
            s_cell_seq, s_hid_seq = tf.scan(
                op_lstm, s_x, initializer=(v_cell, v_hid))
        return s_hid_seq if t_axis == 0 else tf.transpose(s_hid_seq, perm)

    def lyr_gru(
            self, name, s_x, hdim,
            axis=-1, t_axis=0, op_linear=ops.lyr_linear):
        '''
        Args:
            name: string
            s_x: input tensor
            hdim: size of hidden layer
            axis: which axis will RNN op get performed on
            t_axis: which axis would be the timeframe
            op_rnn: RNN layer function, defaults to ops.lyr_gru
        '''
        x_shp = s_x.get_shape().as_list()
        ndim = len(x_shp)
        assert -ndim <= axis < ndim
        assert -ndim <= t_axis < ndim
        axis = axis % ndim
        t_axis = t_axis % ndim
        assert axis != t_axis
        # make sure t_axis is 0, to make scan work
        perm = []
        if t_axis != 0:
            if axis == 0:
                axis = t_axis % ndim
            perm = list(range(ndim))
            perm[0], perm[t_axis] = perm[t_axis], perm[0]
            s_x = tf.transpose(s_x, perm)
        x_shp[t_axis], x_shp[0] = x_shp[0], x_shp[t_axis]
        idim = x_shp[axis]
        assert isinstance(idim, int)
        h_shp = copy.copy(x_shp[1:])
        h_shp[axis-1] = hdim
        with tf.variable_scope(name):
            zero_init = tf.constant_initializer(0.)
            v_cell = tf.get_variable(
                dtype=hparams.FLOATX,
                shape=h_shp, name='cell',
                trainable=False,
                initializer=zero_init)
            self.s_states_di[v_cell.name] = v_cell

            init_range = 0.1 / sqrt(hdim)
            op_gru = lambda _h, _x: ops.lyr_gru_flat(
                'GRU', _x, _h[0],
                axis=axis-1, op_linear=op_linear,
                w_init=tf.random_uniform_initializer(
                    -init_range, init_range, dtype=hparams.FLOATX))
            s_cell_seq, = tf.scan(
                op_gru, s_x, initializer=(v_cell,))
        return s_cell_seq if t_axis == 0 else tf.transpose(s_cell_seq, perm)

    def set_learn_rate(self, lr):
        global g_sess
        g_sess.run(tf.assign(self.v_learn_rate, lr))

    def get_learn_rate(self):
        return g_sess.run(self.v_learn_rate)

    def save_params(self, filename, step=None):
        global g_sess
        save_dir = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(g_sess,
                        filename,
                        global_step=step)

    def load_params(self, filename):
        # if not os.path.exists(filename):
            # stdout.write('Parameter file "%s" does not exist\n' % filename)
            # return False
        self.saver.restore(g_sess, filename)
        return True

    def build(self):
        # create sub-modules
        encoder = hparams.get_encoder()(
            self, 'encoder')
        # ===================
        # build the model

        input_shape = [
            hparams.BATCH_SIZE,
            hparams.MAX_N_SIGNAL,
            None,
            hparams.FEATURE_SIZE]

        s_src_signals = tf.placeholder(
            hparams.COMPLEXX,
            input_shape,
            name='source_signal')
        s_dropout_keep = tf.placeholder(
            hparams.FLOATX,
            [], name='dropout_keep')
        reger = hparams.get_regularizer()
        with tf.variable_scope('global', regularizer=reger):
            # TODO add mixing coeff ?

            # get mixed signal
            s_mixed_signals = tf.reduce_sum(
                s_src_signals, axis=1)

            s_src_signals_pwr = tf.abs(s_src_signals)
            s_mixed_signals_phase = tf.atan2(
                tf.imag(s_mixed_signals), tf.real(s_mixed_signals))
            s_mixed_signals_power = tf.abs(s_mixed_signals)
            s_mixed_signals_log = tf.log1p(s_mixed_signals_power)
            # int[B, T, F]
            # float[B, T, F, E]
            s_embed = encoder(s_mixed_signals_log)
            s_embed_flat = tf.reshape(
                s_embed,
                [hparams.BATCH_SIZE, -1, hparams.EMBED_SIZE])

            # TODO make attractor estimator a submodule ?
            estimator = hparams.get_estimator(
                hparams.TRAIN_ESTIMATOR_METHOD)(self, 'train_estimator')
            s_attractors = estimator(
                s_embed,
                s_src_pwr=s_src_signals_pwr,
                s_mix_pwr=s_mixed_signals_power)

            using_same_method = (
                hparams.INFER_ESTIMATOR_METHOD ==
                hparams.TRAIN_ESTIMATOR_METHOD)

            if using_same_method:
                s_valid_attractors = s_attractors
            else:
                valid_estimator = hparams.get_estimator(
                    hparams.INFER_ESTIMATOR_METHOD
                )(self, 'infer_estimator')
                assert not valid_estimator.USE_TRUTH
                s_valid_attractors = valid_estimator(s_embed)

            separator = hparams.get_separator(
                hparams.SEPARATOR_TYPE)(self, 'separator')
            s_separated_signals_pwr = separator(
                s_mixed_signals_power, s_attractors, s_embed_flat)

            if using_same_method:
                s_separated_signals_pwr_valid = s_separated_signals_pwr
            else:
                s_separated_signals_pwr_valid = separator(
                    s_mixed_signals_power, s_valid_attractors, s_embed_flat)

            # use mixture phase and estimated power to get separated signal
            s_mixed_signals_phase = tf.expand_dims(s_mixed_signals_phase, 1)
            s_separated_signals = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr)

            # loss and SNR for training
            # s_train_loss, v_perms, s_perm_sets = ops.pit_mse_loss(
                # s_src_signals_pwr, s_separated_signals_pwr)
            s_train_loss, v_perms, s_perm_sets = ops.pit_mse_loss(
                s_src_signals, s_separated_signals)

            # resolve permutation
            s_perm_idxs = tf.stack([
                tf.tile(
                    tf.expand_dims(tf.range(hparams.BATCH_SIZE), 1),
                    [1, hparams.MAX_N_SIGNAL]),
                tf.gather(v_perms, s_perm_sets)], axis=2)
            s_perm_idxs = tf.reshape(
                s_perm_idxs, [hparams.BATCH_SIZE*hparams.MAX_N_SIGNAL, 2])
            s_separated_signals = tf.gather_nd(
                s_separated_signals, s_perm_idxs)
            s_separated_signals = tf.reshape(
                s_separated_signals, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FEATURE_SIZE])

            s_train_snr = tf.reduce_mean(ops.batch_snr(
                s_src_signals, s_separated_signals))

            # ^ for validation / inference
            s_valid_loss, v_perms, s_perm_sets = ops.pit_mse_loss(
                s_src_signals_pwr, s_separated_signals_pwr_valid)
            s_perm_idxs = tf.stack([
                tf.tile(
                    tf.expand_dims(tf.range(hparams.BATCH_SIZE), 1),
                    [1, hparams.MAX_N_SIGNAL]),
                tf.gather(v_perms, s_perm_sets)],
                axis=2)
            s_perm_idxs = tf.reshape(
                s_perm_idxs, [hparams.BATCH_SIZE*hparams.MAX_N_SIGNAL, 2])
            s_separated_signals_pwr_valid_pit = tf.gather_nd(
                s_separated_signals_pwr_valid, s_perm_idxs)
            s_separated_signals_pwr_valid_pit = tf.reshape(
                s_separated_signals_pwr_valid_pit, [
                    hparams.BATCH_SIZE,
                    hparams.MAX_N_SIGNAL,
                    -1, hparams.FEATURE_SIZE])

            s_separated_signals_valid = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr_valid_pit,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr_valid_pit)
            s_separated_signals_infer = tf.complex(
                tf.cos(s_mixed_signals_phase) * s_separated_signals_pwr_valid,
                tf.sin(s_mixed_signals_phase) * s_separated_signals_pwr_valid)
            s_valid_snr = tf.reduce_mean(ops.batch_snr(
                s_src_signals, s_separated_signals_valid))


        # ===============
        # prepare summary
        # TODO add impl & summary for word error rate
        with tf.name_scope('train_summary'):
            s_loss_summary_t = tf.summary.scalar('loss', s_train_loss)
            s_snr_summary_t = tf.summary.scalar('SNR', s_train_snr)
            s_lr_summary_t = tf.summary.scalar('LR', self.v_learn_rate)

        with tf.name_scope('valid_summary'):
            s_loss_summary_v = tf.summary.scalar('loss', s_valid_loss)
            s_snr_summary_v = tf.summary.scalar('SNR', s_valid_snr)
            s_lr_summary_v = tf.summary.scalar('LR', self.v_learn_rate)

        # apply optimizer
        ozer = hparams.get_optimizer()(
            learn_rate=self.v_learn_rate, lr_decay=hparams.LR_DECAY)

        v_params_li = tf.trainable_variables()
        r_apply_grads = ozer.compute_gradients(s_train_loss, v_params_li)
        if hparams.GRAD_CLIP_THRES is not None:
            r_apply_grads = [(tf.clip_by_value(
                g, -hparams.GRAD_CLIP_THRES, hparams.GRAD_CLIP_THRES), v)
                for g, v in r_apply_grads if g is not None]
        self.op_sgd_step = ozer.apply_gradients(r_apply_grads)

        self.op_init_params = tf.variables_initializer(v_params_li)
        self.op_init_states = tf.variables_initializer(
            list(self.s_states_di.values()))

        self.train_feed_keys = [
            s_src_signals, s_dropout_keep]
        train_summary = tf.summary.merge(
            [s_loss_summary_t, s_snr_summary_t, s_lr_summary_t])
        self.train_fetches = [
            train_summary,
            dict(loss=s_train_loss, SNR=s_train_snr, LR=self.v_learn_rate),
            self.op_sgd_step]

        self.valid_feed_keys = self.train_feed_keys
        valid_summary = tf.summary.merge([s_loss_summary_v, s_snr_summary_v, s_lr_summary_v])
        self.valid_fetches = [
            valid_summary,
            dict(loss=s_valid_loss, SNR=s_valid_snr)]

        self.infer_feed_keys = [s_mixed_signals, s_dropout_keep]
        self.infer_fetches = dict(signals=s_separated_signals_infer)

        if hparams.DEBUG:
            self.debug_feed_keys = [s_src_signals, s_dropout_keep]
            self.debug_fetches = dict(
                embed=s_embed,
                attrs=s_attractors,
                input=s_src_signals,
                output=s_separated_signals)
            self.debug_fetches.update(encoder.debug_fetches)
            self.debug_fetches.update(separator.debug_fetches)
            if estimator is not None:
                self.debug_fetches.update(estimator.debug_fetches)

        self.saver = tf.train.Saver(var_list=v_params_li)


    def train(self, n_epoch, dataset):
        global g_args
        train_writer = tf.summary.FileWriter(os.path.join(hparams.SUMMARY_DIR, str(datetime.datetime.now().strftime("%m%d_%H%M%S")) + ' ' + hparams.SUMMARY_TITLE), g_sess.graph)
        best_loss = float('+inf')
        best_loss_time = 0
        self.set_learn_rate(hparams.LR)
        print('Set learning rate to %f' % hparams.LR)
        train_step = 0
        valid_step = 0
        for i_epoch in range(n_epoch):
            cli_report = OrderedDict()
            i_batch=0
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'train',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL, shuffle=True)):
                spectra = np.reshape(
                    data_pt[0], [
                        hparams.BATCH_SIZE,
                        hparams.MAX_N_SIGNAL,
                        -1, hparams.FEATURE_SIZE])
                if hparams.MAX_TRAIN_LEN is not None:
                    if spectra.shape[2] > hparams.MAX_TRAIN_LEN:
                        beg = randint(
                            0, spectra.shape[2] - hparams.MAX_TRAIN_LEN-1)
                        spectra = spectra[:, :, beg:beg+hparams.MAX_TRAIN_LEN]
                to_feed = dict(
                    zip(self.train_feed_keys, (
                        spectra, hparams.DROPOUT_KEEP_PROB)))
                step_summary, step_fetch = g_sess.run(
                    self.train_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary, train_step)
                train_step += 1
                stdout.write(':')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            if hparams.LR_DECAY_TYPE == 'adaptive':
                if cli_report['loss'] < best_loss:
                    best_loss = cli_report['loss']
                    best_loss_time = 0
                else:
                    best_loss_time += 1
            elif hparams.LR_DECAY_TYPE == 'fixed':
                best_loss_time += 1
            elif hparams.LR_DECAY_TYPE is None:
                pass
            else:
                raise ValueError(
                    'Unknown LR_DECAY_TYPE "%s"' % hparams.LR_DECAY_TYPE)

            if best_loss_time == hparams.NUM_EPOCH_PER_LR_DECAY:
                best_loss_time = 0
                old_lr = self.get_learn_rate()
                new_lr = old_lr * hparams.LR_DECAY
                self.set_learn_rate(new_lr)
                stdout.write('[LR %f -> %f]' % (old_lr, new_lr))
                stdout.flush()

            if not g_args.no_save_on_epoch:
                if any(map(isnan, cli_report.values())):
                    if i_epoch:
                        stdout.write(
                            '\nEpoch %d/%d got NAN values, restoring last checkpoint ... ')
                        stdout.flush()
                        i_epoch -= 1
                        # FIXME: this path don't work windows
                        self.load_params(
                            'saves/' + self.name + ('_e%d' % (i_epoch+1)))
                        stdout.write('done')
                        stdout.flush()
                        continue
                    else:
                        stdout.write('\nRun into NAN during 1st epoch, exiting ...')
                        sys.exit(-1)
                self.save_params('saves/' + self.name + ('_e%d' % (i_epoch+1)))
                stdout.write('S')
            stdout.write('\nEpoch %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()
            if g_args.no_valid_on_epoch:
                continue
            cli_report = OrderedDict()
            i_batch = 0
            for i_batch, data_pt in enumerate(dataset.epoch(
                    'valid',
                    hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL,
                    shuffle=False)):
                # note: this disables dropout during validation
                to_feed = dict(
                    zip(self.train_feed_keys, (
                        np.reshape(
                            data_pt[0], [
                                hparams.BATCH_SIZE,
                                hparams.MAX_N_SIGNAL,
                                -1, hparams.FEATURE_SIZE]),
                        1.)))
                step_summary, step_fetch = g_sess.run(
                    self.valid_fetches, to_feed)[:2]
                self.reset_state()
                train_writer.add_summary(step_summary, valid_step)
                valid_step+=1
                stdout.write('.')
                stdout.flush()
                _dict_add(cli_report, step_fetch)
            _dict_mul(cli_report, 1. / (i_batch+1))
            stdout.write('\nValid  %d/%d %s\n' % (
                i_epoch+1, n_epoch, _dict_format(cli_report)))
            stdout.flush()

    def test(self, dataset, subset='test', name='Test'):
        global g_args
        train_writer = tf.summary.FileWriter(
            os.path.join(hparams.SUMMARY_DIR,
                         str(datetime.datetime.now().strftime("%m%d_%H%M%S")) + ' ' + hparams.SUMMARY_TITLE), g_sess.graph)
        cli_report = {}
        for data_pt in dataset.epoch(
                subset, hparams.BATCH_SIZE * hparams.MAX_N_SIGNAL):
            # note: this disables dropout during test
            to_feed = dict(
                zip(self.train_feed_keys, (
                    np.reshape(data_pt[0], [hparams.BATCH_SIZE, hparams.MAX_N_SIGNAL, -1, hparams.FEATURE_SIZE]),
                    1.)))
            step_summary, step_fetch = g_sess.run(
                self.valid_fetches, to_feed)[:2]
            train_writer.add_summary(step_summary)
            stdout.write('.')
            stdout.flush()
            _dict_add(cli_report, step_fetch)
        stdout.write(name + ': %s\n' % (
            _dict_format(cli_report)))

    def reset(self):
        '''re-initialize parameters, resets timestep'''
        g_sess.run(tf.global_variables_initializer())

    def reset_state(self):
        '''reset RNN states'''
        g_sess.run([self.op_init_states])

    def parameter_count(self):
        '''
        Returns: integer
        '''
        v_vars_li = tf.trainable_variables()
        return sum(
            reduce(int.__mul__, v.get_shape().as_list()) for v in v_vars_li)
