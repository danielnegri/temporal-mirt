import numpy as np
import scipy.special
import scipy.optimize
from collections import defaultdict

"""
TODO:
- code to initialize abilities at MAP values.
- code to initialize TMIRT training using MIRT parameters.
- check autocorrelation of sampler.
- Write code to show by name the resources that will be most useful to improve performance on an exercise or group of exercises.  Make sure that these are sensible.  Assuming this is good, this will be directly relevant to the recommender engine.
- Visualize the relationship between resources and between exercises captured by the TMIRT, using PCA first, and then something like tSNE.
- Add code to update the abilities vector based on the delay between a student's interactions with resources.  My plan basically is to treat the time delay as another resource which is scaled by the duration of the time delay.  In slightly more detail, I intend to treat the time delay as a biased Gaussian diffusion, with the parameters of the diffusion learned.  This requires some math knowledge, but I could explain it pretty clearly.

- Rotate a to fourier transform for HMC, so can scale different temporal frequencies differently?
"""

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
        else:
            print "unknown resource type"
            print row
            raise StandardError("Unknown resource type")

        # validation
        if self.time_taken < 1.:
            self.time_taken = 1.
        if self.time_taken > 1000.:
            self.time_taken = 1000.


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
            self.a = parent.model.users.a.copy()
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

            if idx is None:
                #idx = np.asarray(range(self.E.shape[0]))
                idx_a = np.arange(parent.model.users.a.shape[1], dtype=int)
            else:
                if idx.shape[0] == 0:
                    # nothing to do
                    return
                # the indices into the abilities matrix corresponding to these
                # users
                compare = np.zeros((
                    np.prod(parent.model.users.a_to_user.shape)), dtype=bool)
                for ii in idx:
                    #print ii
                    #print parent.model.a_to_user.shape
                    #print compare.shape
                    #print np.prod(parent.model.a_to_user.shape)
                    compare += (
                        parent.model.users.a_to_user.reshape((-1)) == ii)
                idx_a = np.nonzero(compare)[0]

            parent.model.users.a[:, idx_a] = self.a[:, idx_a].copy()
            parent.v[:, idx_a] = self.v[:, idx_a].copy()
            parent.Ev[idx] = self.Ev[idx]
            parent.E[idx] = self.E[idx]
            parent.dE[:, idx_a] = self.dE[:, idx_a]

    def __init__(self, model, epsilon=0.1, L=10, beta=0.5):
        self.model = model
        self.epsilon = epsilon
        self.L = L
        self.beta = beta
        self.v = np.random.randn(
            model.users.a.shape[0], model.users.a.shape[1])
        self.E, self.dE = model.E_dE_abilities()
        self.calc_Ev()

    def calc_Ev(self):
        """ calculate the momentum contribution to the energy """
        self.Ev = self.model.map_energy_abilities_to_users(
            np.sum(0.5 * self.v**2, axis=0))

    def leapfrog(self):
        """ integrate leapfrog dynamics for self.L steps """
        a_shape = self.model.users.a.shape

        for _ in range(self.L):
            # initial half step
            self.v -= (
                self.epsilon * self.dE.reshape((-1))).reshape(a_shape)/2.
            # full step in position
            self.model.users.a += (
                self.epsilon * self.v.reshape((-1))).reshape(a_shape)
            self.E, self.dE = self.model.E_dE_abilities()
            # final half step
            self.v -= (
                self.epsilon * self.dE.reshape((-1))).reshape(a_shape)/2.

        # flip the momentum
        self.v = -self.v

        # add the momentum terms to the energy
        self.calc_Ev()

    def sample(self, N=1):
        """ run N steps of Hamiltonian Monte Carlo sampling """
        nacc = 0
        nrej = 0
        for nn in range(N):
            print (
                "sample step %d, acc %d, rej %d, E(x) %g, E(v) %g,"
                " E(x) + E(v) %g" % (nn, nacc, nrej, np.mean(self.E),
                np.mean(self.Ev), np.mean(self.E)
                + np.mean(self.Ev)))
            Sinit = self.state(self)
            self.leapfrog()
            Sleap = self.state(self)
            # calculate the acceptance probabilty for each user
            p_acc = np.exp(Sinit.E - Sleap.E + Sinit.Ev - Sleap.Ev)
            # set the rejected updates back to their original value
            bd = np.nonzero(
                p_acc < np.random.rand(self.E.shape[0], 1).ravel())[0]

            #bd = np.nonzero(Sinit.E < np.inf)[0] # DEBUG

            Sinit.apply_state(bd)
            # flip the momentum
            self.v = -self.v

            nrej += bd.shape[0]
            nacc += self.E.shape[0] - bd.shape[0]

            #self.calc_Ev() # DEBUG
            print (
                "sample step %d, acc %d, rej %d, E(x) %g, E(v) %g, "
                "E(x) + E(v) %g" % (nn, nacc, nrej, np.mean(self.E),
                np.mean(self.Ev), np.mean(self.E)+np.mean(self.Ev)))
            # corrupt the momentum with noise
            self.v = (
                np.sqrt(1.-self.beta)*self.v + np.sqrt(self.beta) *
                np.random.randn(self.v.shape[0], self.v.shape[1]))
            # update the momentum contribution to the energy
            self.calc_Ev()


class TMIRT_users(object):
    """
    Holds user history data for TMIRT.  Note that this history data is specific
    to a single trained TMIRT object, and will not behave sensibly if used with
    a different TMIRT object.
    """

    def __init__(self, model):
        # the parent TMIRT model
        self.model = model

        # holds the maximum number of timesteps for any user
        self.longest_user = 0
        self.num_times_a = 0
        self.num_times_x = 0
        self.num_users = 0
        # provides the index of the user each column in the
        # abilities matrix a belongs to
        self.a_to_user = []
        # for holding exercise correctness and incorrectness
        self.x_correct = []
        # for holding log response time
        self.x_logtime = []
        # holds an array of indices per key
        self.index_lookup = defaultdict(list)

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

            idx_resource = self.model.get_resource_index(r)
            self.index_lookup[('a pre resource', idx_resource)].append(
                                                        self.num_times_a)
            if r.type == 'exercise':
                idx_exercise = self.model.get_exercise_index(r)
                self.index_lookup[('exercise a', idx_exercise)].append(
                                                            self.num_times_a)
                # store the correctness value and response time in an array
                self.x_correct.append(r.correct)
                self.x_logtime.append(np.log(r.time_taken))
                self.index_lookup[('exercise x', idx_exercise)].append(
                                                            self.num_times_x)
                self.num_times_x += 1
            self.increment_num_times(self.num_users)
            self.index_lookup[('a post resource', idx_resource)].append(
                                                            self.num_times_a)

        # store the ending abilities entry for this user
        self.index_lookup['chain end'].append(self.num_times_a)

        # and move on to the next user
        self.increment_num_times(self.num_users)
        self.num_users += 1

    def finalize_users(self):
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

        # turn exercise correctness and time into a numpy array
        self.x_correct = np.asarray(self.x_correct).T
        self.x_logtime = np.asarray(self.x_logtime).T

        # initialize abilities
        self.a = np.random.randn(self.model.num_abilities, self.num_times_a)


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

    To evaluate model on new data:
        call reset_users() to clear old user data
        call users.add_user() for each user history you wish to add
        call users.finalize_users() when all users have been added
        call sample_abilities_HMC_natgrad() to sample abilities estimates
        call predict_performance() to return a matrix of the predicted
        performance of every user on every exercise
    """

    def __init__(self, num_abilities):
        self.num_abilities = num_abilities

        # for holding indices
        self.num_exercises = 0
        self.num_resources = 0
        # holds a single index into the parameter tensors per key, where a key
        # is a unique identifier for a resource or exercise
        self.resource_index = {}
        self.exercise_index = {}

        self.finalized = False

        self.reset_users()

    def predict_user_exercise_performance(self, user_idx, ex_name):
        return self.p_pred[
            self.exercise_index[('exercise', ex_name)], user_idx]


    def predict_performance_once(self, N_avg_samp=1, hmc_steps=10, hmc_burnin=100):
        """
        Called by predict_performance.  Generates the predicted performance of
        every student on every exercise for the current abilities state.
        """

        # get the most recent abilities vector for each student
        a = self.users.a[:, self.users.index_lookup['chain end']]
        # add on a unit to act as a bias
        a = np.vstack((a, np.ones((1, a.shape[1]))))
        Wa = np.dot(self.W_exercise_correct, a)
        return 1./(1. + np.exp(-Wa))


    def predict_performance(self, N_avg_samp=1, hmc_steps=10, hmc_burnin=100, MAP_descent_steps=100):
        """
        Returns a matrix of the predicted performance of every user on every
        exercise.
        Exercises are sorted in the order of their index, as provided by
        exercise_index. Users are sorted by the order in which they were added.

        avg_samp - specifies the number of samples to average the probabilities
        over.
        hmc_steps - specifies how many HMC sampling steps to use between
        evaluation of probabilities.
        MAP_descent_steps - How many LBFGS steps to take to initialize the abilities at their MAP values.
            Set to 0 to disable.
        """

        if MAP_descent_steps > 0:
            new_abilities, L, _ = scipy.optimize.fmin_l_bfgs_b(
                self.E_dE_abilities,
                self.users.a.ravel(),
                disp=1,
                maxfun=MAP_descent_steps, m=10)
            self.users.a = new_abilities.reshape(self.users.a.shape)

        self.sample_abilities_HMC_natgrad(num_steps=hmc_burnin)
        self.p_pred = 0.
        for ii in range(N_avg_samp):
            self.sample_abilities_HMC_natgrad(num_steps=hmc_steps)
            self.p_pred += self.predict_performance_once()/float(N_avg_samp)

        return self.p_pred


    def reset_users(self):
        self.users = TMIRT_users(self)

    def finalize_users(self):
        self.users.finalize_users()

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
            assert(not self.finalized)
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
        # NOTE -- be careful changing the key structure.  it is hard coded elsewhere
        # in the file, and also in load_mirt_parameters in temporal_mirt_train.py

        # return the index, or assign the next available index and return that
        if not (key in self.exercise_index):
            assert(not self.finalized)
            self.exercise_index[key] = self.num_exercises
            self.num_exercises += 1

        return self.exercise_index[key]

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
                            self.W_exercise_correct.flat,
                            self.W_exercise_logtime.flat,
                            self.sigma_exercise_logtime.flat))

    def unflatten_parameters(self, theta):
        theta = self.pop_parameter(theta, self.Phi)
        theta = self.pop_parameter(theta, self.J)
        theta = self.pop_parameter(theta, self.W_exercise_correct)
        theta = self.pop_parameter(theta, self.W_exercise_logtime)
        theta = self.pop_parameter(theta, self.sigma_exercise_logtime)

        # DEBUG TODO(jascha)
        # should enforce positive symmetric definite. this can
        # blow up if a J ever becomes negative
        # enforce symmetric J
        for ii in range(self.num_resources):
            self.J[:, :, ii] = (self.J[:, :, ii] + self.J[:, :, ii].T)/2.

    def map_energy_abilities_to_users(self, E_sub, idx_pre=None):
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

        E = np.zeros((self.users.num_users))

        # we can't do this directly, because Python indexing doesn't handle
        # incrementing the way we would like when the same index occurs
        # multiple times
        #E[idx_pre] += E_sub

        # set the stride long enough the same index never occurs twice in the
        # same indexing array on the left side of the assignment
        # (remembering that idx_pre is sorted)
        if idx_pre==None:
            for ii in range(self.users.longest_user):
                E[self.users.a_to_user[ii::self.users.longest_user]] \
                    += E_sub[ii::self.users.longest_user]
        else:
            for ii in range(self.users.longest_user):
                E[self.users.a_to_user[idx_pre[ii::self.users.longest_user]]] \
                    += E_sub[ii::self.users.longest_user]

        # DEBUG
        #for ii in range(E_sub.shape[0]):
        #    E[self.a_to_user[idx_pre[ii]]] \
        #                += E_sub[ii]
        return E

    def E_chain_start(self, Ea=None):
        """ the energy contribution from t=1 (before any resource) per user """
        idx = self.users.index_lookup['chain start']
        a = self.users.a[:, idx]
        E = 0.5 * np.sum(a**2, axis=0)
        Ea[idx] += E
        return

    def E_dEda_accumulate_chain_start(self, da, Ea):
        """ the energy contribution from t=1 (before any resource) per user """
        idx = self.users.index_lookup['chain start']
        a = self.users.a[:, idx]
        da[:, idx] += a
        E = 0.5 * np.sum(a**2, axis=0)
        Ea[idx] += E
        return

    def E_exercise(self, idx_exercise, Ea=None):
        """
        The energy contribution from the conditional distribution over the
        exercise identified by idx_exercise.
        """
        x_correct, x_time, err, idx_x, a, idx_a, Wa_correct, Wa_time, sigma = \
            self.get_exercise_matrices(idx_exercise)
        E = np.log(1. + np.exp(-x_correct*(Wa_correct)))
        E += err**2 / sigma**2 / 2.
        E += 0.5 * np.log(sigma**2)
        Ea[idx_a] += E
        return

    def dEdtheta_exercise(self, idx_exercise):
        """ The derivative of the energy function, summed over all users,
        for a single exercise. """
        x_correct, x_time, err, idx_x, a, idx_a, Wa_correct, Wa_time, sigma = \
            self.get_exercise_matrices(idx_exercise)

        expxWa = np.exp(-x_correct*(Wa_correct))
        term1 = (-1./(1. + expxWa)*expxWa*x_correct)
        dEdW_correct = np.dot(term1, a.T)

        dEderr = err / sigma**2
        dEdW_time = np.dot(dEderr.reshape((1, -1)), a.T)
        #print dEderr.shape, a.shape

        dEdsigma_time =  np.sum(-err ** 2) / sigma ** 3
        dEdsigma_time += np.max(err.shape) / sigma  # 0.5 log sigma**2 term

        return dEdW_correct, dEdW_time, dEdsigma_time

    def E_dEda_accumulate_exercise(self, idx_exercise, da, Ea):
        """ The derivative of the energy function in terms of a,
        for all users, for a single exercise. """
        x_correct, x_time, err, idx_x, a, idx_a, Wa_correct, Wa_time, sigma = \
            self.get_exercise_matrices(idx_exercise)

        expxWa = np.exp(-x_correct*(Wa_correct))
        term1 = (-1./(1. + expxWa)*expxWa*x_correct)
        W = self.W_exercise_correct[idx_exercise, :-1]
        da[:, idx_a] += np.dot(W.reshape((-1, 1)), term1.reshape((1, -1)))

        W_time = self.W_exercise_logtime[idx_exercise, :-1]
        dEderr = err / sigma**2
        dEda_time = np.dot(W_time.reshape((-1, 1)), dEderr.reshape((1,-11)))
        da[:, idx_a] += dEda_time

        E = np.log(1. + expxWa)
        E += err**2 / sigma**2 / 2.
        E += 0.5 * np.log(sigma**2)
        Ea[idx_a] += E
        return

    def E_resource(self, idx_resource, Ea=None):
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
        # dangerous to think of the energy here as corresponding to a single
        # column of a

        Ea[idx_pre] += E
        return

    def E_dEda_accumulate_resource(self, idx_resource, da, Ea):
        idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)
        Phi = self.Phi[:, :, idx_resource]

        da[:, idx_pre] += (
            -np.dot(Phi.T, np.dot(J, a_err))[:-1, :] - np.dot(J, a_err))
        da[:, idx_post] += np.dot(J, a_err)

        E = 0.5 * np.sum(a_err*np.dot(J, a_err), axis=0)
        # DEBUG check sign
        # Gaussian normalization term
        E += -np.sum(np.log(np.real(np.linalg.eig(J)[0])))  # *idx_pre.shape[0]
        # NOTE adjacent columns in a are coupled, and that
        # coupling energy is only assigned to one of the columns, so it's
        # dangerous to think of the energy here as corresponding to a single
        # column of a
        Ea[idx_pre] += E
        return

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
        if ('exercise x', idx_exercise) in self.users.index_lookup:
            idx_x = self.users.index_lookup[('exercise x', idx_exercise)]
            idx_a = self.users.index_lookup[('exercise a', idx_exercise)]
        else:
            idx_x = np.array([], dtype=int)
            idx_a = np.array([], dtype=int)
        x = self.users.x_correct[idx_x]
        # add on a unit to act as a bias
        a = np.vstack((self.users.a[:, idx_a], np.ones((1, idx_a.shape[0]))))

        # make correctness in {-1,1}
        x = 2*x - 1

        W_correct = self.W_exercise_correct[[idx_exercise], :]
        Wa_correct = np.dot(W_correct, a).reshape((-1))
        W_time = self.W_exercise_logtime[[idx_exercise], :]
        Wa_time = np.dot(W_time, a).reshape((-1))

        x_time = self.users.x_logtime[idx_x]
        err = (Wa_time - x_time)

        sigma = self.sigma_exercise_logtime[idx_exercise]

        return x, x_time, err, idx_x, a, idx_a, Wa_correct, Wa_time, sigma

    def get_abilities_matrices(self, idx_resource):
        # get the pre and post abilities matrices that correspond to this
        # resource
        if ('a pre resource', idx_resource) in self.users.index_lookup:
            idx_pre = self.users.index_lookup[('a pre resource', idx_resource)]
            idx_post = self.users.index_lookup[('a post resource', idx_resource)]
        else:
            idx_pre = np.array([], dtype=int)
            idx_post = np.array([], dtype=int)
        # add on a unit to act as a bias
        a_pre = np.vstack((self.users.a[:, idx_pre], np.ones((1, idx_pre.shape[0]))))
        a_post = self.users.a[:, idx_post]
        # get the parameters for this resource
        Phi = self.Phi[:, :, idx_resource]
        J = self.J[:, :, idx_resource]
        # do the actual computation
        a_predicted = a_pre[:-1, :] + np.dot(Phi, a_pre)
        a_err = a_post - a_predicted
        return idx_pre, idx_post, a_pre, a_post, a_err, J

    def E(self, a=None):
        """
        accumulate the energy for each column in the abilities vector
        """

        # TODO(jascha) is this ever called with a not equal None?
        # (leftover from Gaussian diffusion sampling?)
        if a is not None:
            a_old = self.users.a
            self.users.a = a

        E = np.zeros((self.users.num_times_a))
        self.E_chain_start(Ea=E)
        for idx_resource in range(self.num_resources):
            self.E_resource(idx_resource, Ea=E)
        for idx_exercise in range(self.num_exercises):
            self.E_exercise(idx_exercise, Ea=E)

        if a is not None:
            self.users.a = a_old

        return E

    def E_dE_abilities(self, a=None):
        """
        Return the energy per user and the energy gradient with respect
        to the abilities matrix.
        """
        # calculate the energy
        #E = self.E()

        if not a == None:
            self.users.a = a.reshape(self.users.a.shape)

        Ea = np.zeros((self.users.num_times_a))
        da = np.zeros(self.users.a.shape)
        self.E_dEda_accumulate_chain_start(da, Ea)
        for idx_resource in range(self.num_resources):
            self.E_dEda_accumulate_resource(idx_resource, da, Ea)
        for idx_exercise in range(self.num_exercises):
            self.E_dEda_accumulate_exercise(idx_exercise, da, Ea)

        E = self.map_energy_abilities_to_users(Ea.ravel())

        if not a == None:
            E = np.sum(E)
            da = da.reshape(a.shape)

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
        dW_exercise_logtime = np.zeros(self.W_exercise_logtime.shape)
        dsigma_exercise_logtime = np.zeros(self.sigma_exercise_logtime.shape)
        for idx_exercise in range(self.num_exercises):
            dW_exercise_correct[idx_exercise, :], dW_exercise_logtime[idx_exercise, :], dsigma_exercise_logtime[idx_exercise] = \
                self.dEdtheta_exercise(idx_exercise)

        dE = np.concatenate((dPhi.flat, dJ.flat, dW_exercise_correct.flat, dW_exercise_logtime.flat, dsigma_exercise_logtime.flat))

        return E/self.users.num_users, dE/self.users.num_users

    def get_joint_gaussian_covariance_bias(self):

        full_J = np.zeros((self.users.num_times_a, self.num_abilities))

        for idx_resource in range(self.num_resources):
            idx_pre, idx_post, a_pre, a_post, a_err, J = \
                self.get_abilities_matrices(idx_resource)
            Phi = self.Phi[:, :, idx_resource]
            phi_m = Phi[:, :-1]  # no bias
            # phi_b = Phi[:, [-1]]  # bias only
            Jpre = np.dot(phi_m.T, np.dot(J, phi_m))
            # Jcross = np.dot(J, phi_m)

            full_J[idx_post, :] += np.diag(J)
            full_J[idx_pre, :] += np.diag(Jpre)

        # and the univariate Gaussian over the initial state
        idx_start = self.users.index_lookup['chain start']
        full_J[idx_start, :] += 1.

        W = full_J.ravel()**(-1./4.)

        # TODO(jascha) set full_bias
        full_bias = np.zeros((self.users.num_times_a, self.num_abilities))

        return W, full_bias

    def sample_abilities_HMC_natgrad(
            self, num_steps=50, epsilon=0.1, L=10, beta=0.5):

        # we will scale our dynamics by W (add ref)
        W, _ = self.get_joint_gaussian_covariance_bias()
        W *= epsilon

        sampler = HMC(self, epsilon=W, L=L, beta=beta)
        sampler.sample(N=int(num_steps))

        return sampler.E

    def finalize_training_data(self):
        """
        - Convert many python lists into numpy arrays now that we know their
         final size.
        - Create the data structures that hold the parameters and abilities.
        """

        self.users.finalize_users()

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
        self.sigma_exercise_logtime = np.ones((self.num_exercises))

        self.finalized = True
