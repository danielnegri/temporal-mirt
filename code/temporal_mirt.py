import numpy as np
from scipy import sparse
import scipy.special
import accuracy_model_util as acc_util
from collections import defaultdict
import sys


class TMIRTResource(object):
    """
    Holds information on a single resource for the temporal mirt model.
    """
    def __init__(self, row, idx_pl):
        if row[idx_pl.rowtype] == 'problemlog':
            self.correct = row[idx_pl.correct]
            self.time_done = row[idx_pl.time_done]
            self.time_taken = row[idx_pl.time_taken]
            self.name = row[idx_pl.exercise]
            self.type = 'exercise'
        elif row[idx_pl.rowtype] == 'exercise':
            # TODO (eliana): Treat videos and exercises differently
            self.correct = 1 if row[idx_pl.correct] == 'True' else 0
            self.time_done = row[idx_pl.time_done]
            self.time_taken = row[idx_pl.time_taken]
            self.name = row[idx_pl.exercise]
            self.type = 'exercise'
        elif row[idx_pl.rowtype] == 'video':
            # TODO (eliana): Treat videos and exercises differently
            self.completed = 1 if row[idx_pl.correct] == 'True' else 0
            self.time_done = row[idx_pl.time_done]
            self.time_taken = row[idx_pl.time_taken]
            self.name = row[idx_pl.exercise]
            self.type = 'video'


class HMC(object):
    """
    A Hamiltonian Monte Carlo sampler.
    This is specifically designed to work with a TMIRT object, which has
    some interesting properties.  Samples should be accepted or rejected
    per-user, but the variables should be stored per time-slice, so the
    sampler needs to be aware of the user <-> time slice mapping.
    Additionally, we want to use the sparse scaling matrix epsilon
    to hide the joint Gaussian covariance structure.
    """

    class state():
        """
        holds all the parameters which are updated by the sampler
        """
        def __init__(self, parent):
            self.a = parent.model.a.copy()
            self.v = parent.v.copy()
            self.E = parent.E.copy()
            self.Ev = parent.Ev.copy()
            self.dE = parent.dE.copy()
            self.parent = parent
        def apply_state(self, idx=None):
            """
            set the current state of the sampler/model to this state, for the
            users listed in idx.  if idx is none, do it for all users.
            """
            parent = self.parent

            if idx == None:
                #idx = np.asarray(range(self.E.shape[0]))
                idx_a = np.arange(parent.model.a.shape[1], dtype=int)
            else:
                if idx.shape[0] == 0:
                    # nothing to do
                    return
                # the indices into the abilities matrix corresponding to these
                # users
                compare = np.zeros((np.prod(parent.model.a_to_user.shape)), dtype=bool)
                for ii in idx:
                    #print ii
                    #print parent.model.a_to_user.shape
                    #print compare.shape
                    #print np.prod(parent.model.a_to_user.shape)
                    compare += (parent.model.a_to_user.reshape((-1)) == ii)
                idx_a = np.nonzero(compare)[0]

            parent.model.a[:,idx_a] = self.a[:,idx_a].copy()
            parent.v[:,idx_a] = self.v[:,idx_a].copy()
            parent.Ev[idx] = self.Ev[idx]
            parent.E[idx] = self.E[idx]
            parent.dE[:,idx_a] = self.dE[:,idx_a]


    def __init__(self, model, epsilon=0.1, L=10, beta=0.5):
        self.model = model
        self.epsilon = epsilon
        self.L = L
        self.beta = beta
        self.v = np.random.randn(model.a.shape[0], model.a.shape[1])
        self.E, self.dE = model.E_dE_abilities()
        self.calc_Ev()


    def calc_Ev(self):
        """ calculate the momentum contribution to the energy """
        idx_pre = np.arange(self.v.shape[1], dtype=int)
        self.Ev = self.model.map_energy_abilities_to_users(np.sum(0.5 * self.v**2, axis=0), idx_pre)


    def leapfrog(self):
        """ integrate leapfrog dynamics for self.L steps """
        a_shape = self.model.a.shape

        for _ in range(self.L):
            # initial half step
            #self.v -= np.dot(self.epsilon, self.dE.reshape((-1,1))).reshape(a_shape)/2.
            #self.v -= self.epsilon.dot(self.dE.reshape((-1,1))).reshape(a_shape)/2.
            self.v -= (self.epsilon * self.dE.reshape((-1))).reshape(a_shape)/2.
            # full step in position
            #self.model.a += self.epsilon.T.dot(self.v.reshape((-1,1))).reshape(a_shape)
            self.model.a += (self.epsilon * self.v.reshape((-1))).reshape(a_shape)
            self.E, self.dE = self.model.E_dE_abilities()
            # final half step
            #self.v -= self.epsilon.dot(self.dE.reshape((-1,1))).reshape(a_shape)/2.
            self.v -= (self.epsilon * self.dE.reshape((-1))).reshape(a_shape)/2.

        # flip the momentum
        self.v = -self.v

        # add the momentum terms to the energy
        self.calc_Ev()


    def sample(self, N=1):
        """ run N steps of Hamiltonian Monte Carlo sampling """
        nacc = 0
        nrej = 0
        for nn in range(N):
            print "sample step %d, acc %d, rej %d, E(x) %g, E(v) %g, E(x) + E(v) %g"%(nn, nacc, nrej, np.mean(self.E), np.mean(self.Ev), np.mean(self.E)+np.mean(self.Ev))
            Sinit = self.state(self)
            self.leapfrog()
            Sleap = self.state(self)
            # calculate the acceptance probabilty for each user
            p_acc = np.exp(Sinit.E - Sleap.E + Sinit.Ev - Sleap.Ev)
            # set the rejected updates back to their original value
            bd = np.nonzero(p_acc < np.random.rand(self.E.shape[0],1).ravel())[0]

            #bd = np.nonzero(Sinit.E < np.inf)[0] # DEBUG

            Sinit.apply_state(bd)
            # flip the momentum
            self.v = -self.v

            nrej += bd.shape[0]
            nacc += self.E.shape[0] - bd.shape[0]

            #self.calc_Ev() # DEBUG
            print "sample step %d, acc %d, rej %d, E(x) %g, E(v) %g, E(x) + E(v) %g"%(nn, nacc, nrej, np.mean(self.E), np.mean(self.Ev), np.mean(self.E)+np.mean(self.Ev))
            # corrupt the momentum with noise
            self.v = np.sqrt(1.-self.beta)*self.v + np.sqrt(self.beta)*np.random.randn(self.v.shape[0],self.v.shape[1])
            # update the momentum contribution to the energy
            self.calc_Ev()


# TODO(jascha) should break this apart into two classes, one to hold the model
# and the other to hold the training data
class TMIRT(object):
    """
    Holds:
      - User data for training a temporal multidimensional item
        response theory (TMIRT) model.
      - The functions for evaluating the model energy and gradients
        for those users.
      - The function for sampling the abilities.
    """

    def __init__(self, num_abilities):
        self.num_abilities = num_abilities

        # for holding indices
        self.num_exercises = 0
        self.num_resources = 0
        self.num_times_a = 0
        self.num_times_x = 0
        self.num_users = 0
        # holds the maximum number of timesteps for any user
        self.longest_user = 0
        # provides the index of the user each column in the
        # abilities matrix a belongs to
        self.a_to_user = []
        # holds a single index into the parameter tensors per key, where a key
        # is a unique identifier for a resource or exercise
        self.resource_index = {}
        self.exercise_index = {}
        # holds an array of indices per key
        self.index_lookup = defaultdict(list)

        # for holding exercise correctness and incorrectness
        self.x = []

        self.sampler = None

    def get_resource_index(self, resource):
        """
        look up or assign the index which corresponds to a given resource

        this is the index into Phi, J
        """

        # make the key
        key = (resource.type, resource.name)
        if resource.type == 'exercise':
            # correct and incorrectly answered problems are treated as
            # separate resources
            key += (resource.correct,)

        # return the index, or assign the next available index and return that
        if not (key in self.resource_index):
            self.resource_index[key] = self.num_resources
            self.num_resources += 1

        return self.resource_index[key]

    def get_exercise_index(self, exercise):
        """
        look up or assign the index which corresponds to a given exercise

        this is the index into W
        """

        # make the key
        key = (exercise.type, exercise.name)

        # return the index, or assign the next available index and return that
        if key in self.exercise_index:
            return self.exercise_index[key]
        else:
            self.exercise_index[key] = self.num_exercises
            self.num_exercises += 1

    def pop_parameter(self, theta, xx):
        """
        pull entries out of the beginning of theta and place them in xx until
         xx is full, then chop off the beginning of theta.  called by
        unflatten_parameters.
        """
        xl = np.prod(xx.shape)
        xx[:] = theta[:xl].reshape(xx.shape)[:]
        return theta[xl:]

    def flatten_parameters(self):
        #return np.concatenate((self.Phi.flat, self.J.flat,
        #    self.W_exercise_correct.flat, self.W_exercise_logtime.flat))
        return np.concatenate((self.Phi.flat, self.J.flat,
                            self.W_exercise_correct.flat))

    def unflatten_parameters(self, theta):
        theta = self.pop_parameter(theta, self.Phi)
        theta = self.pop_parameter(theta, self.J)
        theta = self.pop_parameter(theta, self.W_exercise_correct)
        #theta = self.pop_parameter(theta, self.W_exercise_logtime)

        # DEBUG should enforce positive symmetric definite. this can
        # blow up if a J ever becomes negative
        # enforce symmetric
        for ii in range(self.num_resources):
            self.J[:, :, ii] = (self.J[:, :, ii] + self.J[:, :, ii].T)/2.

    def map_energy_abilities_to_users(self, E_sub, idx_pre):
        """
        takes an input with a contribution to the energy for each column in the
        a matrix.

        outputs an energy for each user by summing over the columns for that
        user.

        (It is important to have a separate energy per user so that the a
        variables can be sampled independently for each user.  Otherwise either
        all the a would need to be updated or none of them -- and for any
        reasonable sampling step size the answer would be none of them.)
        """

        E = np.zeros((self.num_users))

        # we can't do this directly, because Python indexing doesn't handle
        # incrementing the way we would like when the same index occurs
        # multiple times
        #E[idx_pre] += E_sub

        # set the stride long enough the same index never occurs twice in the
        # same indexing array on the left side of the assignment
        # (remembering that idx_pre is sorted)
        for ii in range(self.longest_user):
            E[self.a_to_user[idx_pre[ii::self.longest_user]]] \
                += E_sub[ii::self.longest_user]

        # DEBUG
        #for ii in range(E_sub.shape[0]):
        #    E[self.a_to_user[idx_pre[ii]]] \
        #                += E_sub[ii]
        return E

    def E_chain_start(self):
        """ the energy contribution from t=1 (before any resource) per user """
        idx = self.index_lookup['chain start']
        a = self.a[:, idx]
        E = 0.5 * np.sum(a**2, axis=0)
        E = self.map_energy_abilities_to_users(E, idx)
        return E

    def dEda_accumulate_chain_start(self, da):
        """ the energy contribution from t=1 (before any resource) per user """
        idx = self.index_lookup['chain start']
        a = self.a[:, idx]
        da[:,idx] += a
        return da

    def E_exercise(self, idx_exercise):
        """
        The energy contribution from the conditional distribution over the
        exercise identified by idx_exercise.
        """
        # TODO(jascha) -- add 'time taken' to x and try to predict it
        x, idx_x, a, idx_a, Wa = self.get_exercise_matrices(idx_exercise)
        E = np.log(1. + np.exp(-x*(Wa)))

        E = self.map_energy_abilities_to_users(E.ravel(), idx_a)

        return E

    def dEdW_exercise(self, idx_exercise):
        """ The derivative of the energy function, summed over all users,
        for a single exercise. """
        x, idx_x, a, idx_a, Wa = self.get_exercise_matrices(idx_exercise)
        expxWa = np.exp(-x*(Wa))
        term1 = (-1./(1. + expxWa)*expxWa*x)
        dEdW = np.dot(term1, a.T)
        return dEdW

    def dEda_accumulate_exercise(self, idx_exercise, da):
        """ The derivative of the energy function in terms of a,
        for all users, for a single exercise. """
        x, idx_x, a, idx_a, Wa = self.get_exercise_matrices(idx_exercise)
        expxWa = np.exp(-x*(Wa))
        term1 = (-1./(1. + expxWa)*expxWa*x)
        W = self.W_exercise_correct[idx_exercise, :-1]
        da[:,idx_a] += np.dot(W.reshape((-1,1)), term1.reshape((1,-1)))

    def E_resource(self, idx_resource):
        """
        the energy contribution from the resource identified by "idx_resource",
        returned per user
        """
        idx_pre, idx_post, a_pre, a_post, a_err, J = \
            self.get_abilities_matrices(idx_resource)

        E = 0.5 * np.sum(a_err*np.dot(J, a_err), axis=0)
        # DEBUG check sign
        # Gaussian normalization term
        E += -np.sum(np.log(np.real(np.linalg.eig(J)[0])))  # *idx_pre.shape[0]

        # NOTE adjacent columns in a are coupled, and that
        # coupling energy is only assigned to one of the columns, so it's
        # dangerous to think of the energy here as corresponding to a single column
        # of a

        E = self.map_energy_abilities_to_users(E, idx_pre)

        return E


    def dEda_accumulate_resource(self, idx_resource, da):
        idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)
        Phi = self.Phi[:, :, idx_resource]

        da[:,idx_pre] += -np.dot(Phi.T, np.dot(J, a_err))[:-1, :] - np.dot(J, a_err)
        da[:,idx_post] += np.dot(J, a_err)

    def dEdPhi_resource(self, idx_resource):
        idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)

        dEdPhi = -np.dot(np.dot(J, a_err), a_pre.T)
        return dEdPhi

    def dEdJ_resource(self, idx_resource):
        idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)

        dEdJ = 0.5*np.dot(a_err, a_err.T)
        dEdJ += -np.linalg.inv(J.T)*idx_pre.shape[0]

        return dEdJ

    def get_exercise_matrices(self, idx_exercise):
        # TODO (eliana): This fails when there's only one example of a given
        # exercise so it's never pre or something
        idx_x = self.index_lookup[('exercise x', idx_exercise)]
        idx_a = self.index_lookup[('exercise a', idx_exercise)]
        x = self.x[:, idx_x]
        a = self.a[:, idx_a]
        # add on a unit to act as a bias
        a = np.vstack((a, np.ones((1, a.shape[1]))))

        # make correctness in {-1,1}
        x = 2*x - 1

        W = self.W_exercise_correct[idx_exercise, :]
        Wa = np.dot(W, a)
        return x, idx_x, a, idx_a, Wa

    def get_abilities_matrices(self, idx_resource):
        # get the pre and post abilities matrices that correspond to this
        # resource

        idx_pre = self.index_lookup[('a pre resource', idx_resource)]
        idx_post = self.index_lookup[('a post resource', idx_resource)]
        a_pre = self.a[:, idx_pre]
        a_post = self.a[:, idx_post]
        # add on a unit to act as a bias
        a_pre = np.vstack((a_pre, np.ones((1, a_pre.shape[1]))))
        # get the parameters for this resource
        Phi = self.Phi[:, :, idx_resource]
        J = self.J[:, :, idx_resource]
        # do the actual computation
        a_predicted = a_pre[:-1, :] + np.dot(Phi, a_pre)
        a_err = a_post - a_predicted
        return idx_pre, idx_post, a_pre, a_post, a_err, J

    def E(self, a=None):
        """
        accumulate the energy for each user
        """

        if a is not None:
            a_old = self.a
            self.a = a

        E = 0.
        E += self.E_chain_start()
        for idx_resource in range(self.num_resources):
            E += self.E_resource(idx_resource)
        for idx_exercise in range(self.num_exercises):
            E += self.E_exercise(idx_exercise)

        if a is not None:
            self.a = a_old

        return E


    def E_dE_abilities(self):
        """
        Return the energy per user and the energy gradient with respect
        to the abilities matrix.
        """
        # calculate the energy
        E = self.E()

        da = np.zeros(self.a.shape)
        self.dEda_accumulate_chain_start(da)
        for idx_resource in range(self.num_resources):
            self.dEda_accumulate_resource(idx_resource, da)
        for idx_exercise in range(self.num_exercises):
            self.dEda_accumulate_exercise(idx_exercise, da)

        # DEBUG check these gradients
        return E, da

    def E_dE(self, theta):
        """
        Update the parameters to theta, and return the total energy and
         (flattened) gradient. For gradient descent of the parameters.
        """
        # set the parameters
        self.unflatten_parameters(theta)
        # calculate the energy
        E = np.sum(self.E())

        dPhi = np.zeros(self.Phi.shape)
        dJ = np.zeros(self.J.shape)
        for idx_resource in range(self.num_resources):
            dPhi[:, :, idx_resource] = self.dEdPhi_resource(idx_resource)
            dJ[:, :, idx_resource] = self.dEdJ_resource(idx_resource)

        dW_exercise_correct = np.zeros(self.W_exercise_correct.shape)
        for idx_exercise in range(self.num_exercises):
            dW_exercise_correct[idx_exercise, :] = \
                self.dEdW_exercise(idx_exercise)

        dE = np.concatenate((dPhi.flat, dJ.flat, dW_exercise_correct.flat))

        return E/self.num_users, dE/self.num_users

    def increment_num_times(self, num_users):
        self.a_to_user.append(num_users)
        self.num_times_a += 1

    def add_user(self, user, resources):
        """
        step through all the resources this user interacted with, in
        chronological order and assign and index each time slice to a
        column in the abilities storage matrix
        """

        self.longest_user = max(self.longest_user, len(resources)+1)

        # store the starting abilities entry for this user
        self.index_lookup['chain start'].append(self.num_times_a)

        for r in resources:

            idx_resource = self.get_resource_index(r)
            self.index_lookup[('a pre resource', idx_resource)].append(
                                                        self.num_times_a)
            if r.type == 'exercise':
                idx_exercise = self.get_exercise_index(r)
                self.index_lookup[('exercise a', idx_exercise)].append(
                                                            self.num_times_a)
                # store the correctness value in an array
                self.x.append(r.correct)
                self.index_lookup[('exercise x', idx_exercise)].append(
                                                            self.num_times_x)
                self.num_times_x += 1
            self.increment_num_times(self.num_users)
            self.index_lookup[('a post resource', idx_resource)].append(
                                                            self.num_times_a)

        self.increment_num_times(self.num_users)

        ## store the starting and ending abilities entry index for this user
        #self.user_index_range.append([start_time, self.num_times_a])

        self.num_users += 1

    def finalize_training_data(self):
        """
        - Convert many python lists into numpy arrays now that we know their
         final size.
        - Create the data structures that hold the parameters and abilities.
        """

        # turn the index arrays into numpy arrays for later fast indexing
        self.index_lookup = dict(self.index_lookup)
        for k in self.index_lookup:
            self.index_lookup[k] = np.asarray(self.index_lookup[k])
        self.a_to_user = np.asarray(self.a_to_user)
        #self.user_index_range = np.asarray(self.user_index_range)

        # turn exercise correctness into a numpy array
        self.x = np.asarray(self.x).T

        # abilities
        self.a = np.random.randn(self.num_abilities, self.num_times_a)
        # parameters
        self.Phi = np.zeros((self.num_abilities,
                             self.num_abilities + 1, self.num_resources))
        self.J = np.tile(np.eye(self.num_abilities).reshape(
                                (self.num_abilities, self.num_abilities, 1)),
                                (1, 1, self.num_resources)) * 10
        self.W_exercise_correct = np.random.randn(self.num_exercises,
                                    self.num_abilities + 1)/np.sqrt(self.num_abilities)
        self.W_exercise_logtime = np.zeros((self.num_exercises,
                                    self.num_abilities + 1))
        # exact zeros cause problems for sparse matrices
        self.J += np.random.randn(*self.J.shape)*1e-10
        self.Phi += np.random.randn(*self.Phi.shape)*1e-10


    def invsqrtm(self, W):
        """
        Numpy/scipy only has limited support for sparse matrices, so we're
        rolling our own function for matrix^(-1/2), using the Taylor
        expansion.
        If we weren't scaling by "mn" below, this would only be valid if all the
        eigenvalues of W had magnitude less than 1.  With the scaling, we need
        all the eigenvalues of W/mn to have magnitude less than 1.
        """

        # number of terms to use in the Taylor series.  increase to get a more
        # accurate approximation.
        # DEBUG -- this is purely heuristic!
        nterms = self.longest_user*10

        # DEBUG experiment with different sparse matrix types.
        # eg, make sure these aren't LIL, which is slower.

        # A will accumulate the output
        A = scipy.special.binom(-0.5, 0)*scipy.sparse.eye(W.shape[0], W.shape[1])
        # B will accumulate the powers of (W-I)
        B = scipy.sparse.eye(W.shape[0], W.shape[1])

        # the closer W is to the identity, the faster this converges,
        # so scale W so it is closer to the identity, and then apply
        # the corresponding scaling to the inverse
        ind = W.shape[0]
        dg = W[ind, ind]
        mn = np.mean(dg)
        W = W/mn

        # the terms in the series are (W-I), not W
        W = W - scipy.sparse.eye(W.shape[0], W.shape[1])
        # accumulate terms in the Taylor series
        for i in range(nterms)+1:
            coeff = scipy.special.binom(-0.5, i)
            B = W.dot(B)
            A += coeff*B

        # scale the square root of the inverse of W to match the scaling of W
        # above
        A /= np.sqrt(mn)

        return A


    def get_joint_gaussian_covariance_bias(self):

        full_J = np.zeros((self.num_times_a, self.num_abilities))

        for idx_resource in range(self.num_resources):
            idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)
            Phi = self.Phi[:, :, idx_resource]
            phi_m = Phi[:,:-1] # no bias
            phi_b = Phi[:,[-1]] # bias only
            Jpre = np.dot(phi_m.T, np.dot(J, phi_m))
            Jcross = np.dot(J, phi_m)

            full_J[idx_post, :] += np.diag(J)
            full_J[idx_pre,  :] += np.diag(Jpre)

        # and the univariate Gaussian over the initial state
        idx_start = self.index_lookup['chain start']
        full_J[idx_start, :] += 1.

#        W = sparse.lil_matrix(
#           (self.num_times_a*self.num_abilities, self.num_times_a*self.num_abilities))
#
#        # DEBUG
#        #W.setdiag(1. / np.sqrt(full_J.ravel()))
#        W.setdiag(full_J.ravel()**(-1./4.))
#
#        W = W.tocsr()

        W = full_J.ravel()**(-1./4.)

        # TODO(jascha) set full_bias
        full_bias = np.zeros((self.num_times_a, self.num_abilities))

        return W, full_bias

        # TODO(jascha)  sparse matrices suck in python!! the below code is insanely slow.
        # figure out a way to make them suck less, or to not use them.

        # full ability to ability coupling matrix
        full_J = sparse.lil_matrix(
           (self.num_times_a*self.num_abilities, self.num_times_a*self.num_abilities))
        # full abilities bias vector
        full_bias = sparse.lil_matrix((self.num_times_a, self.num_abilities))

        # accumulate terms in the coupling matrix and bias vector for all resources
        for idx_resource in range(self.num_resources):
            idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)
            Phi = self.Phi[:, :, idx_resource]
            phi_m = Phi[:,:-1] # no bias
            phi_b = Phi[:,[-1]] # bias only
            Jpre = np.dot(phi_m.T, np.dot(J, phi_m))
            Jcross = np.dot(J, phi_m)

            print idx_resource, idx_pre.shape, idx_post.shape


            # TODO these loops are horribly inefficient!  some kind of wrapper for
            # N-d sparse matrices?  Expanding the idx arrays?
            for ai in range(self.num_abilities):
                for aj in range(self.num_abilities):
                    # python defaults to row major indexing
                    full_J[idx_post + self.num_times_a+ai, idx_post + self.num_times_a+aj] += J[ai,aj]*np.ones((len(idx_post)))
                    full_J[idx_pre + self.num_times_a+ai, idx_pre + self.num_times_a+aj] += Jpre[ai,aj]*np.ones((len(idx_pre)))
                    full_J[idx_post + self.num_times_a+ai, idx_pre + self.num_times_a+aj] += Jcross[ai,aj]*np.ones((len(idx_post)))
                    full_J[idx_pre + self.num_times_a+ai,idx_post + self.num_times_a+aj] += Jcross.T[ai,aj]*np.ones((len(idx_pre)))
            # DEBUG check for factor of 2
            full_bias[idx_post,:] += (np.dot(J, phi_b)).T
            full_bias[idx_pre,:] += (np.dot(phi_m.T, np.dot(J, phi_b))).T

        # TODO convert from LIL to ?


        W = invsqrtm(full_J)
        # DEBUG(jascha) haven't checked this bias is correct, but not yet using it 
        # for anything.
        full_bias = W.dot(W.dot(full_bias))
        return W, full_bias


    def sample_abilities_HMC_natgrad(self,num_steps=1e3,epsilon=0.1,L=10,beta=0.5):

        # we will scale our dynamics by W (add ref)
        W, _ = self.get_joint_gaussian_covariance_bias()
        W *= epsilon

        #if self.sampler == None:
        #    self.sampler = HMC(self, epsilon=W, L=L, beta=beta)
        #self.sampler.sample(N=num_steps)

        sampler = HMC(self, epsilon=W, L=L, beta=beta)
        sampler.sample(N=num_steps)

        return sampler.E

 
    def sample_abilities_diffusion(self, num_steps=1e4, epsilon=None):

        if epsilon is None:
            epsilon = (0.1 / np.sqrt(self.num_abilities)) / \
                    np.sqrt(self.longest_user)

        # calculate the energy for the initialization state
        E_current = self.E()
        for i in range(num_steps):
            # generate the proposal state
            a_proposed = self.a + epsilon * np.random.randn(self.num_abilities,
                                                            self.num_times_a)

            E_proposed = self.E(a=a_proposed)

            # this is required to avoid overflow when E_abilities - E_proposal
            # is very large
            idx = E_current > E_proposed
            E_current[idx] = E_proposed[idx]

            # probability of accepting proposal
            p_accept = np.exp(E_current - E_proposed)

            if np.mean(p_accept) < 0.1:
                print >>sys.stderr, "low sampling accept rate mean", np.mean(p_accept), "by sample", p_accept

            #assert np.isfinite(E_proposed), "non-finite proposal energy"

            pcmp = np.random.rand(p_accept.shape[0], 1).reshape(p_accept.shape)
            update_idx = np.nonzero(p_accept > pcmp)[0]
            self.a[:, update_idx] = a_proposed[:, update_idx]
            E_current[:, update_idx] = E_proposed[:, update_idx]

        return E_current
