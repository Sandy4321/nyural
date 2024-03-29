"""Implementation of recurrent network, including Gauss-Newton approximation
for use in Hessian-free optimization.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J., & Sutskever, I. (2011). Learning recurrent neural networks with
hessian-free optimization. Proceedings of the 28th International Conference on
Machine Learning.
"""

import numpy as np
from ffnet import FFNet


class RNNet(FFNet):
    def __init__(self, shape, struc_damping=None, rec_layers=None,
                 W_rec_params={}, truncation=None, **kwargs):
        """Initialize the parameters of the network.

        :param struc_damping: controls scale of structural damping (relative
            to Tikhonov damping)
        :param rec_layers: by default, all layers except the first and last
            are recurrently connected. A list of booleans can be passed here
            to override that on a layer-by-layer basis.
        :param W_rec_params: weight initialization parameter dict for recurrent
            weights (passed to init_weights, see parameter descriptions there)
        :param truncation: a tuple (n,k) where backpropagation through time
            will be executed every n timesteps and run backwards for k steps
            (defaults to full backprop if None)

        See FFNet for the rest of the parameters.
        """

        # define recurrence for each layer (needs to be done before super
        # constructor because this is used in compute_offsets)
        if rec_layers is None:
            # assume all recurrent except first/last layer
            rec_layers = [False] + [True] * (len(shape) - 2) + [False]

        if len(rec_layers) != len(shape):
            raise ValueError("Must define recurrence for each layer")

        self.rec_layers = rec_layers

        # super constructor
        super(RNNet, self).__init__(shape, **kwargs)

        self.struc_damping = struc_damping
        self.truncation = truncation

        # add on recurrent weights
        if kwargs.get("load_weights", None) is None and np.any(rec_layers):
            self.W = np.concatenate(
                (self.W, self.init_weights([(self.shape[l], self.shape[l])
                                            for l in range(self.n_layers)
                                            if rec_layers[l]],
                                           **W_rec_params)))

    def forward(self, input, params, deriv=False, init_activations=None,
                init_state=None):
        """Compute activations for given input sequence and parameters.

        If deriv=True then also compute the derivative of the activations.

        If init_activations/init_state are set then they will be used to
        initialize the recurrent inputs or the nonlinearity states,
        respectively.
        """

        # input shape = [batch_size, seq_len, input_dim]
        # activations shape = [n_layers, batch_size, seq_len, layer_size]

        if callable(input):
            # reset the plant
            # TODO: allow the initial state of plant to be set?
            input.reset()
        elif input.ndim < 3:
            # then we've just been given a single sample (rather than batch)
            input = input[None, ...]

        activations = [np.zeros((input.shape[0], input.shape[1], l),
                                dtype=self.dtype)
                       for l in self.shape]

        if deriv:
            d_activations = [None for l in self.layers]

        for i, l in enumerate(self.layers):
            # reset any state in the nonlinearities
            l.reset(None if init_state is None else init_state[i])

        W_recs = [self.get_weights(params, (i, i))
                  for i in np.arange(self.n_layers)]
        for s in np.arange(input.shape[1]):
            for i in np.arange(self.n_layers):
                if i == 0:
                    # get the external input
                    if callable(input):
                        # call the plant with the output of the previous
                        # timestep to generate the next input
                        if s == 0 and init_activations is not None:
                            ff_input = input(init_activations[-1])
                        else:
                            # note: this will pass zeros on the first timestep
                            # if init_activations is not set
                            ff_input = input(activations[-1][:, s - 1])
                    else:
                        ff_input = input[:, s]
                else:
                    # compute feedforward input
                    ff_input = np.zeros_like(activations[i][:, s])
                    for pre in self.back_conns[i]:
                        W, b = self.get_weights(params, (pre, i))
                        ff_input += np.dot(activations[pre][:, s], W) + b

                # recurrent input
                if self.rec_layers[i]:
                    if s > 0:
                        rec_input = np.dot(activations[i][:, s - 1],
                                           W_recs[i][0])
                    elif init_activations is None:
                        # apply bias input on first timestep
                        rec_input = W_recs[i][1]
                    else:
                        # use the provided activations to initialize the
                        # 'previous' timestep
                        rec_input = np.dot(init_activations[i],
                                           W_recs[i][0])
                else:
                    rec_input = 0

                # apply activation function
                activations[i][:, s] = (
                    self.layers[i].activation(ff_input + rec_input))

                # compute derivative
                if deriv:
                    d_act = self.layers[i].d_activation(ff_input + rec_input,
                                                        activations[i][:, s])
                    d_act = d_act[:, None, :]
                    if d_activations[i] is None:
                        d_activations[i] = d_act
                    else:
                        d_activations[i] = np.concatenate((d_activations[i],
                                                           d_act), axis=1)

        if deriv:
            return activations, d_activations

        return activations

    def calc_grad(self):
        """Compute parameter gradient."""

        grad = np.zeros_like(self.W)
        W_recs = [self.get_weights(self.W, (l, l))
                  for l in np.arange(self.n_layers)]

        sig_len = self.inputs.shape[1]
        if self.truncation is None:
            trunc_per = trunc_len = sig_len
        else:
            trunc_per, trunc_len = self.truncation

        for n in np.arange(trunc_per - 1, sig_len, trunc_per):
            # every trunc_per timesteps we want to run backprop

            deltas = [np.zeros((self.inputs.shape[0], l), dtype=self.dtype)
                      for l in self.shape]
            state_deltas = [None if not l.stateful else
                            np.zeros((self.inputs.shape[0], self.shape[i]),
                                     dtype=self.dtype)
                            for i, l in enumerate(self.layers)]

            # backpropagate error
            for s in np.arange(n, np.maximum(n - trunc_len, -1), -1):
                # execute trunc_len steps of backprop through time

                for l in np.arange(self.n_layers - 1, -1, -1):
                    if l == self.n_layers - 1:
                        # derivative of loss
                        error = self.loss[1](self.activations[-1][:, s],
                                             self.targets[:, s])
                    else:
                        # error from feedforward weights
                        error = np.zeros_like(deltas[l])
                        for post in self.conns[l]:
                            c_error = np.dot(deltas[post],
                                             self.get_weights(self.W,
                                                              (l, post))[0].T)
                            error += c_error

                            # feedforward gradient
                            offset, W_end, b_end = self.offsets[(l, post)]
                            grad[offset:W_end] += self.outer_sum(
                                self.activations[l][:, s]
                                if self.GPU_activations is None else
                                [l, np.index_exp[:, s]], deltas[post])
                            grad[W_end:b_end] += np.sum(deltas[post], axis=0)

                    # add recurrent error
                    if self.rec_layers[l]:
                        error += np.dot(deltas[l], W_recs[l][0].T)

                    # compute deltas
                    if not self.layers[l].stateful:
                        deltas[l] = self.J_dot(self.d_activations[l][:, s],
                                               error, transpose=True)
                    else:
                        d_input = self.d_activations[l][:, s, ..., 0]
                        d_state = self.d_activations[l][:, s, ..., 1]
                        d_output = self.d_activations[l][:, s, ..., 2]

                        state_deltas[l] += self.J_dot(d_output, error,
                                                      transpose=True)
                        deltas[l] = self.J_dot(d_input, state_deltas[l],
                                               transpose=True)
                        state_deltas[l] = self.J_dot(d_state, state_deltas[l],
                                                     transpose=True)

                    # gradient for recurrent weights
                    if self.rec_layers[l]:
                        offset, W_end, b_end = self.offsets[(l, l)]
                        if s > 0:
                            grad[offset:W_end] += self.outer_sum(
                                self.activations[l][:, s - 1]
                                if self.GPU_activations is None else
                                [l, np.index_exp[:, s - 1]], deltas[l])
                        else:
                            # put remaining gradient into initial bias
                            grad[W_end:b_end] += np.sum(deltas[l], axis=0)

        # divide by batchsize
        grad /= self.inputs.shape[0]

        return grad

    def check_grad(self, calc_grad):
        """Check gradient via finite differences (for debugging)."""

        eps = 1e-6
        grad = np.zeros(calc_grad.shape)

        sig_len = self.inputs.shape[1]
        if self.truncation is None:
            trunc_per = trunc_len = sig_len
        else:
            trunc_per, trunc_len = self.truncation

        inc_W = np.zeros_like(self.W)



        for n in np.arange(trunc_per, sig_len + 1, trunc_per):
            start = np.maximum(n - trunc_len, 0)

            # the truncated backprop gradient is equivalent to running the
            # network normally for the initial timesteps and then just changing
            # the parameters for the truncation period.  so that's what we're
            # simulating here.
            if start > 0:
                prev = self.forward(self.inputs[:, :start], self.W)
                init_a = [p[:, -1] for p in prev]
                init_s = [l.state.copy() if l.stateful else None
                          for l in self.layers]
            else:
                init_a = None
                init_s = None

            for i in np.arange(len(self.W)):
                inc_W[:] = 0
                inc_W[i] = eps

                out_inc = self.forward(self.inputs[:, start:n], self.W + inc_W,
                                       init_activations=init_a,
                                       init_state=init_s)
                out_dec = self.forward(self.inputs[:, start:n], self.W - inc_W,
                                       init_activations=init_a,
                                       init_state=init_s)

                error_inc = self.loss[0](out_inc[-1], self.targets[:, start:n])
                error_inc = np.sum(error_inc) / len(error_inc)

                error_dec = self.loss[0](out_dec[-1], self.targets[:, start:n])
                error_dec = np.sum(error_dec) / len(error_dec)

                grad[i] += (error_inc - error_dec) / (2 * eps)

        try:
            assert np.allclose(calc_grad, grad, rtol=1e-3)
        except AssertionError:
            print "calc_grad"
            print calc_grad
            print "finite grad"
            print grad
            print "calc_grad - finite grad"
            print calc_grad - grad
            print "calc_grad / finite grad"
            print calc_grad / grad
            raw_input("Paused (press enter to continue)")

    def calc_G(self, v, damping=0):
        """Compute Gauss-Newton matrix-vector product."""

        Gv = np.zeros(self.W.size, dtype=self.dtype)

        sig_len = self.inputs.shape[1]

        # R forward pass
        R_states = [None if not l.stateful else
                    np.zeros((self.inputs.shape[0], self.shape[i]),
                             dtype=self.dtype)
                    for i, l in enumerate(self.layers)]
        R_activations = [np.zeros_like(a) for a in self.activations]
        v_recs = [self.get_weights(v, (l, l))
                  for l in np.arange(self.n_layers)]
        W_recs = [self.get_weights(self.W, (l, l))
                  for l in np.arange(self.n_layers)]

        for s in np.arange(sig_len):
            for l in np.arange(self.n_layers):
                R_inputs = np.zeros((self.inputs.shape[0], self.shape[l]),
                                    dtype=self.dtype)
                # input from feedforward connections
                if l > 0:
                    for pre in self.back_conns[l]:
                        vw, vb = self.get_weights(v, (pre, l))
                        Ww, _ = self.get_weights(self.W, (pre, l))

                        R_inputs += np.dot(self.activations[pre][:, s],
                                           vw) + vb
                        R_inputs += np.dot(R_activations[pre][:, s], Ww)

                # recurrent input
                if self.rec_layers[l]:
                    if s == 0:
                        # bias input on first step
                        R_inputs += v_recs[l][1]
                    else:
                        R_inputs += np.dot(self.activations[l][:, s - 1],
                                            v_recs[l][0])
                        R_inputs += np.dot(R_activations[l][:, s - 1],
                                           W_recs[l][0])

                if not self.layers[l].stateful:
                    R_activations[l][:, s] = (
                        self.J_dot(self.d_activations[l][:, s], R_inputs))
                else:
                    d_input = self.d_activations[l][:, s, ..., 0]
                    d_state = self.d_activations[l][:, s, ..., 1]
                    d_output = self.d_activations[l][:, s, ..., 2]

                    R_states[l] = self.J_dot(d_state, R_states[l])
                    R_states[l] += self.J_dot(d_input, R_inputs)
                    R_activations[l][:, s] = self.J_dot(d_output, R_states[l])

        # R backward pass
        if self.truncation is None:
            trunc_per = trunc_len = sig_len
        else:
            trunc_per, trunc_len = self.truncation

        for n in np.arange(trunc_per - 1, sig_len, trunc_per):
            R_deltas = [np.zeros((self.inputs.shape[0], l), dtype=self.dtype)
                        for l in self.shape]
            R_states = [np.zeros_like(R_deltas[i]) if l.stateful else None
                        for i, l in enumerate(self.layers)]

            for s in np.arange(n, np.maximum(n - trunc_len, -1), -1):
                for l in np.arange(self.n_layers - 1, -1, -1):
                    if l == self.n_layers - 1:
                        # output layer
                        R_error = (R_activations[-1][:, s] *
                                   self.loss[2](self.activations[l][:, s],
                                                self.targets[:, s]))
                    else:
                        # error from feedforward connections
                        R_error = np.zeros_like(self.activations[l][:, s])
                        for post in self.conns[l]:
                            W, _ = self.get_weights(self.W, (l, post))
                            R_error += np.dot(R_deltas[post], W.T)

                            # feedforward gradient
                            offset, W_end, b_end = self.offsets[(l, post)]
                            Gv[offset:W_end] += self.outer_sum(
                                self.activations[l][:, s]
                                if self.GPU_activations is None else
                                [l, np.index_exp[:, s]], R_deltas[post])
                            Gv[W_end:b_end] += np.sum(R_deltas[post], axis=0)

                    # add recurrent error
                    if self.rec_layers[l]:
                        R_error += np.dot(R_deltas[l], W_recs[l][0].T)

                    # compute deltas
                    if not self.layers[l].stateful:
                        R_deltas[l] = self.J_dot(self.d_activations[l][:, s],
                                                 R_error, transpose=True)
                    else:
                        d_input = self.d_activations[l][:, s, ..., 0]
                        d_state = self.d_activations[l][:, s, ..., 1]
                        d_output = self.d_activations[l][:, s, ..., 2]

                        R_states[l] += self.J_dot(d_output, R_error,
                                                  transpose=True)
                        R_deltas[l] = self.J_dot(d_input, R_states[l],
                                                 transpose=True)
                        R_states[l] = self.J_dot(d_state, R_states[l],
                                                 transpose=True)

                    # apply structural damping
                    struc_damping = getattr(self.optimizer, "struc_damping",
                                            None)
                    if struc_damping is not None and self.rec_layers[l]:
                        # penalize change in the output w.r.t. input (which is
                        # what R_activations represents)
                        R_deltas[l] += struc_damping * R_activations[l][:, s]

                    # recurrent gradient
                    if self.rec_layers[l]:
                        offset, W_end, b_end = self.offsets[(l, l)]
                        if s > 0:
                            Gv[offset:W_end] += self.outer_sum(
                                self.activations[l][:, s - 1]
                                if self.GPU_activations is None else
                                [l, np.index_exp[:, s - 1]], R_deltas[l])
                        else:
                            Gv[W_end:b_end] += np.sum(R_deltas[l], axis=0)

        Gv /= self.inputs.shape[0]

        Gv += damping * v  # Tikhonov damping

        return Gv

    def calc_J(self, start=0, stop=None):
        """Compute the Jacobian of the network via finite differences."""

        eps = 1e-6
        N = self.W.size

        # as in check_grad, the truncation is equivalent to running the network
        # normally for the initial timesteps and then changing the parameters,
        # so that's what we do here to compute the Jacobian

        if start > 0:
            prev = self.forward(self.inputs[:, :start], self.W)
            init_a = [p[:, -1] for p in prev]
            init_s = [l.state.copy() if l.stateful else None
                      for l in self.layers]
        else:
            init_a = None
            init_s = None

        if stop is None:
            stop = self.inputs.shape[1]

        # compute the Jacobian
        J = None
        for i in range(N):
            inc_i = np.zeros(N)
            inc_i[i] = eps

            inc = self.forward(self.inputs[:, start:stop], self.W + inc_i,
                               init_activations=init_a, init_state=init_s)[-1]
            dec = self.forward(self.inputs[:, start:stop], self.W - inc_i,
                               init_activations=init_a, init_state=init_s)[-1]

            if start > 0:
                inc = np.concatenate((prev[-1], inc), axis=1)
                dec = np.concatenate((prev[-1], dec), axis=1)

            J_i = (inc - dec) / (2 * eps)
            if J is None:
                J = J_i[..., None]
            else:
                J = np.concatenate((J, J_i[..., None]), axis=-1)

        return J

    def check_G(self, calc_G, v, damping=0):
        """Check Gv calculation via finite differences (for debugging)."""

        sig_len = self.inputs.shape[1]
        if self.truncation is None:
            trunc_per = trunc_len = sig_len
        else:
            trunc_per, trunc_len = self.truncation

        G = np.zeros((len(self.W), len(self.W)))
        Gv = np.zeros(len(self.W))

        for n in np.arange(trunc_per, sig_len + 1, trunc_per):
            start = np.maximum(n - trunc_len, 0)

            # compute Jacobian
            # note that we do a full forward pass and a partial backwards
            # pass, so we only truncate the backwards J matrix
            J = self.calc_J(0, n)
            trunc_J = self.calc_J(start, n) if start > 0 else J

            # second derivative of loss function
            L = self.loss[2](self.activations[-1][:, :n],
                             self.targets[:, :n])
            # TODO: check self.loss[2] via finite differences

            G += np.einsum("abji,abj,abjk->ik", trunc_J, L, J)

        # divide by batch size
        G /= self.inputs.shape[0]

        Gv = np.dot(G, v)
        Gv += damping * v

        try:
            assert np.allclose(calc_G, Gv, rtol=1e-3)
        except AssertionError:
            print "calc_G"
            print calc_G
            print "finite G"
            print Gv
            print "calc_G - finite G"
            print calc_G - Gv
            print "calc_G / finite G"
            print calc_G / Gv
            raw_input("Paused (press enter to continue)")

    def compute_offsets(self):
        """Precompute offsets for layers in the overall parameter vector."""

        ff_offset = super(RNNet, self).compute_offsets()

        # offset for recurrent weights is end of ff weights
        offset = ff_offset
        for l in np.arange(self.n_layers):
            if self.rec_layers[l]:
                self.offsets[(l, l)] = (
                    offset,
                    offset + self.shape[l] * self.shape[l],
                    offset + (self.shape[l] + 1) * self.shape[l])
                offset += (self.shape[l] + 1) * self.shape[l]

        return offset - ff_offset
