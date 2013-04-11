import numpy as np
import accuracy_model_util as acc_util
from collections import defaultdict


class TMIRTResource(object):
    """
    Holds information on a single resource for the temporal mirt model.
    """
    def __init__(self, row):
        # used to index the fields with a line of text in the input data file
        idx_pl = acc_util.FieldIndexer(acc_util.FieldIndexer.plog_fields)
        if row[idx_pl.rowtype] == 'problemlog':
            self.correct = row[idx_pl.correct]
            self.time_done = row[idx_pl.time_done]
            self.time_taken = row[idx_pl.time_taken]
            self.name = row[idx_pl.exercise]
            self.type = 'exercise'


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
        a matrix (note, adjacent columns in a will be coupled, and that
        coupling energy is only assigned to one of the columns, so it's
        dangerous to think of the input energy corresponding to a single column
        of a).

        outputs an energy for each user by summing over the columns for that
        user.

        (It is important to have a separate energy per user so that the a
        variables can be sampled independently for each user.  Otherwise either
        all the a would need to be updated or none of them -- and for any
        reasonable sampling step size the answer would be none of them.)
        """

        # we can't do this directly, because Python indexing doesn't handle
        # incrementing the way we would like when the same index occurs
        # multiple times
        #E[idx_pre] += E_sub

        # TODO(jascha) There's probably a more efficient way to do this

        E = np.zeros((self.num_users))
        # set the stride long enough the same index never occurs twice in the
        # same indexing array on the left side of the assignment
        # (remembering that idx_pre is sorted)
        # DEBUG
        for ii in range(E_sub.shape[0]):
            E[self.a_to_user[idx_pre[ii]]] \
                        += E_sub[ii]
        #for ii in range(self.longest_user):
        #    E[self.a_to_user[idx_pre[ii::self.longest_user]]] \
        #                += E_sub[ii::self.longest_user]

        return E

    def E_chain_start(self):
        """ the energy contribution from t=1 (before any resource) per user """
        idx = self.index_lookup['chain start']
        a = self.a[:, idx]
        E = 0.5 * np.sum(a**2, axis=0)
        E = self.map_energy_abilities_to_users(E, idx)
        return E

    def E_exercise(self, idx_exercise):
        """
        The energy contribution from the conditional distribution over the
        exercise identified by idx_exercise.
        """
        # TODO(jascha) -- add 'time taken' to x and try to predict it

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

        E = np.log(1. + np.exp(-x*(Wa)))

        E = self.map_energy_abilities_to_users(E.ravel(), idx_a)

        return E

    def dEdW_exercise(self, idx_exercise):
        """ The derivative of the energy function, summed over all users,
        for a single exercise. """
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
        expxWa = np.exp(-x*(Wa))
        term1 = (-1./(1. + expxWa)*expxWa*x)
        dEdW = np.dot(term1, a.T)
        return dEdW

    def E_resource(self, idx_resource):
        """
        the energy contribution from the resource identified by "idx_resource",
        returned per user
        """
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

        E = 0.5 * np.sum(a_err*np.dot(J, a_err), axis=0)
        # DEBUG check sign
        # Gaussian normalization term
        E += -np.sum(np.log(np.linalg.eig(J)[0])) #*idx_pre.shape[0]

        E = self.map_energy_abilities_to_users(E, idx_pre)

        return E

    def dEdPhi_resource(self, idx_resource):
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
        dEdPhi = -np.dot(np.dot(J, a_err), a_pre.T)
        return dEdPhi

    def dEdJ_resource(self, idx_resource):
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
        
        dEdJ = 0.5*np.dot(a_err, a_err.T)
        dEdJ += -np.linalg.inv(J.T)*idx_pre.shape[0]

        return dEdJ

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

        return E, dE

    def increment_num_times(self, num_users):
        self.num_times_a += 1
        self.a_to_user.append(num_users)

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
                                (1, 1, self.num_resources)) * 100
        self.W_exercise_correct = np.zeros((self.num_exercises,
                                    self.num_abilities + 1))
        self.W_exercise_logtime = np.zeros((self.num_exercises,
                                    self.num_abilities + 1))

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

            #assert np.isfinite(E_proposed), "non-finite proposal energy"

            pcmp = np.random.rand(p_accept.shape[0],1).reshape(p_accept.shape)
            update_idx = np.nonzero(p_accept > pcmp)[0]
            self.a[:,update_idx] = a_proposed[:,update_idx]
            E_current[:,update_idx] = E_proposed[:,update_idx]

        return E_current

"""
    def logL_AIS(self, num_steps=1e3, epsilon=None):
        ""
        Calculate the joint log probability of the training data, using AIS

        NOT YET IMPLEMENTED.  This is just cut and paste from the mirt code.
        ""

        # initialize abilities using prior
        abilities = np.random.randn(num_abilities, 1)

        # the prior and joint energy function at this location
        E0n = 0.5 * np.sum(abilities**2) - num_abilities/2. * np.log(2.*np.pi)
        ENn = (0.5 * np.sum(abilities**2) - num_abilities/2. * np.log(2.*np.pi) +
               np.sum(-np.log(mirt_util.conditional_probability_observed(
                        abilities, couplings,
                        exercises_ind, correct))))

        # initialize the weights
        logw = E0n
        for n in range(1,num_steps):
            mix_frac1 = float(n+1)/num_steps
            mix_frac0 = 1.-mix_frac1
            Elast = mix_frac0*E0n  + mix_frac1*ENn;

            # update the sample and the energy
            abilities = mirt_util.sample_abilities_diffusion(
                couplings, exercises_ind, correct,
                abilities_init=abilities, num_steps=1,
                sampling_epsilon=sampling_epsilon,
                conditional_weight=mix_frac1)[0]

            # the prior and joint energy function at this location
            E0n = 0.5 * np.sum(abilities**2) - num_abilities/2. * np.log(2.*np.pi)
            ENn = (0.5 * np.sum(abilities**2) - num_abilities/2. * np.log(2.*np.pi) +
                   np.sum(-np.log(mirt_util.conditional_probability_observed(
                            abilities, couplings,
                            exercises_ind, correct))))

            Enew = mix_frac0*E0n  + mix_frac1*ENn

            # accumulate the change in log probability from this intermediate
            # distribution
            logw += Enew - Elast

        logw -= ENn

        return logw
        """
