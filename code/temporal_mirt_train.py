"""This script trains, and then emits features generated using,
a temporal multidimensional item response theory model.
"""

import argparse
import datetime
import fileinput
import numpy as np
import sys
import scipy
import scipy.optimize
import random

import accuracy_model_util as acc_util

from temporal_mirt import TMIRT, TMIRTResource

# used to index the fields in with a line of text in the input data file
linesplit = acc_util.linesplit


def get_cmd_line_arguments():
    parser = argparse.ArgumentParser(
        description="Train a temporal multi-dimensional item response theory"
        " (TMIRT) model.")
    parser.add_argument("-a", "--num_abilities", type=int, default=2,
           help="Number of hidden ability units.")
    parser.add_argument("-s", "--sampling_num_steps", type=int, default=10,
        help="Number of sampling steps to use for the E step.")
    parser.add_argument("-l", "--sampling_epsilon", type=float, default=0.1,
     help="The length scale to use for sampling update proposals.")
    parser.add_argument("-n", "--num_epochs", type=int, default=10000,
        help="The number of EM iterations to do during learning")
    parser.add_argument("-q", "--num_replicas", type=int, default=1,
        help="""The number of copies of the data to train on. If there is too little
                training data, increase this number in order to maintain multiple samples
                from the abilities vector for each student.  A sign that there is too
                little training data is if the update step length ||dcouplings|| remains
                large.""")
    parser.add_argument("-m", "--max_pass_lbfgs", type=int, default=5,
        help="The number of LBFGS descent steps to make per M step.")
    parser.add_argument("-e", "--min_training_examples", type=int, default=5,
        help="""The minimum number of resources a user must have used to be
                included in the training dataset.""")
    # The weight for an L2 regularizer on the parameters.  This can be very
    # small, but keeps the weights from running away in a weakly constrained
    # direction.
    #parser.add_argument("-p", "--regularization", default=None) #1e-5)
    #DEBUG not currently implemented

    #parser.add_argument("-u", "--ais_users", type=int, default=100,
    #  help="Number of users to use to compute log likelihood.")
    parser.add_argument("-v", "--ais_steps", type=int, default=1e5,
        help="""Maximum number of intermediate distributions (sampling steps)
        to use when computing log likelihood via AIS.  If the estimate doesn't
        converge, then increase this number.""")
    parser.add_argument("-f", "--file", type=str,
            default='data/user_assessment.responses',
            help="The source data file.")
    parser.add_argument("-o", "--output", type=str, default='',
        help="The root filename for output.")
    parser.add_argument("-i", "--indexer", type=str, default='plog',
        help="This defines the model that you'll use to parse the data\
        (look in accuracy_model_util for examples")
    parser.add_argument("-b", "--bootstrap_mirt", type=str, default='',
        help="Initialize the parameters using the mIRT model with the given name.")
    parser.add_argument("--bootstrap_epochs", type=int, default=5,
        help="How many training epochs to fix the MIRT values for parameters\
        shared by MIRT and TMIRT models.")

    # DEBUG use parse_known_args rather than parse_args so can easily run it
    # inside pylab
    options, _ = parser.parse_known_args()

    if options.output == '':
        # default filename
        file_start = options.file.rfind('/')
        options.output = "%stmirt_file=%s_abilities=%d" % (
                options.file[:file_start + 1],
                options.file[file_start + 1:], options.num_abilities)

    return options


def log_likelihood_AIS(user_states, couplings, options):
    """ taking from mirt code.  not yet modified for tmirt. """

    max_steps = options.ais_steps
    step_counts = np.ceil(np.logspace(0, np.log(max_steps),
                                      np.ceil(np.log(max_steps)), base=np.e))
    step_counts = step_counts.astype(int)

    print "(# steps) log L, min log L"

    for num_steps in step_counts:
        log_L = 0.
        min_log_L = 0.
        for state in user_states:
            # TODO(eliana)
            # This isn't used - should it be
            # abilities = state['abilities'].copy()
            correct = state['correct']
            exercises_ind = state['exercises_ind']
            #print "pre", num_steps
            log_L += log_probability_singleuser_AIS(couplings, exercises_ind,
                                correct,
                                num_steps=num_steps,
                                sampling_epsilon=options.sampling_epsilon)
            min_log_L += np.log(0.5)*correct.shape[0]
            #print "post", log_L
        log_L /= len(user_states)
        min_log_L /= len(user_states)
        # if this series doesn't converge, then you need more steps!
        print "(%d steps) %f, %f" % (num_steps, log_L, min_log_L)


def load_data(options):
    """
    Create a new TMIRT model, and load it with training data.
    """

    model = TMIRT(options.num_abilities)

    prev_user = None
    resources = []

    idx_pl = acc_util.get_FieldIndexer(options.indexer)

    print >>sys.stderr, "loading data"
    # loop through the data multiple times if we want multiple copies of the
    # data in the training set
    for _ in range(options.num_replicas):
        # loop through all the training data, and create user objects
        for line in fileinput.input(options.file):
            # split on either tab or \x01 so the code works via Hive or pipe
            row = linesplit.split(line.strip())
            # the user and timestamp are shared by all row types.
            # load the user
            user = row[idx_pl.user]
            if user != prev_user:
                # We're getting a new user, so perform the reduce operation
                # on our previous user
                if len(resources) > options.min_training_examples:
                    model.users.add_user(user, resources)
                resources = []
            prev_user = user
            if row[idx_pl.rowtype] == 'problemlog':
                row[idx_pl.correct] = row[idx_pl.correct] == 'true'
                row[idx_pl.eventually_correct] = (
                    row[idx_pl.eventually_correct] == 'true')
                row[idx_pl.problem_number] = int(row[idx_pl.problem_number])
                row[idx_pl.number_attempts] = int(row[idx_pl.number_attempts])
                row[idx_pl.number_hints] = int(row[idx_pl.number_hints])
                row[idx_pl.time_taken] = float(row[idx_pl.time_taken])
            resources.append(TMIRTResource(row, idx_pl))

        if len(resources) > options.min_training_examples:
            # flush the data for the final user, too
            model.users.add_user(user, resources)
        resources = []

        fileinput.close()
    # create parameter structures, change datatypes, etc
    model.finalize_training_data()

    return model


def check_gradients_E_step():
    """
    test gradients by:
    import temporal_mirt_train
    temporal_mirt_train.check_gradients_E_step()
    """
    options = get_cmd_line_arguments()
    print >>sys.stderr, "Checking gradients E step.", options  # DEBUG

    step_size = 1e-6
    model = load_data(options)

    a0 = model.users.a.copy()
    f0, df0 = model.E_dE_abilities()
    # test gradients in a random order.  This lets us run check gradients on
    # the full size model, but still statistically test every type of gradient.
    while True:
        ind0 = np.floor(np.random.rand()*a0.shape[0])
        ind1 = np.floor(np.random.rand()*a0.shape[1])

        model.users.a = a0.copy()
        model.users.a[ind0, ind1] += step_size
        f1, df1 = model.E_dE_abilities()

        df_true = np.sum((f1 - f0))/step_size

        print (
            "ind", ind0, ind1, "df pred", df0[ind0, ind1], "df true", df_true,
            "df pred - df true", df0[ind0, ind1] - df_true)


def check_gradients_M_step():
    """
    test gradients by:
    import temporal_mirt_train
    temporal_mirt_train.check_gradients_M_step()
    """
    options = get_cmd_line_arguments()
    print >>sys.stderr, "Checking gradients M step.", options  # DEBUG

    step_size = 1e-6

    model = load_data(options)

    # index lookup so we know where the problem is
    Phi_l = np.prod(model.Phi.shape)
    J_l = np.prod(model.J.shape)
    W_ex_cr_l = np.prod(model.W_exercise_correct.shape)
    s_ex_tm_l = model.W_exercise_correct.shape[0]
    print (
        "Phi %d-%d" % (0, Phi_l), "J %d-%d" % (Phi_l, Phi_l+J_l),
        "W_ex_cr %d-%d" % (Phi_l+J_l, Phi_l+J_l+W_ex_cr_l),
        "W_ex_tm %d-%d" % (Phi_l+J_l+W_ex_cr_l, Phi_l+J_l+2*W_ex_cr_l),
        "sigma_ex_tm %d-%d" % (Phi_l+J_l+2*W_ex_cr_l, Phi_l+J_l+2*W_ex_cr_l+s_ex_tm_l),
        )

    theta = model.flatten_parameters()
    f0, df0 = model.E_dE(theta)
    # test gradients in a random order.  This lets us run check gradients on
    # the full size model, but still statistically test every type of gradient.
    test_order = range(theta.shape[0])
    random.shuffle(test_order)
    for ind in test_order:
        if ind < Phi_l:
            print "Phi ",
        elif ind < Phi_l+J_l:
            print "J   ",
        elif ind < Phi_l+J_l+W_ex_cr_l:
            print "W_cr",
        elif ind < Phi_l+J_l+2*W_ex_cr_l:
            print "W_tm",
        elif ind < Phi_l+J_l+2*W_ex_cr_l+s_ex_tm_l:
            print "s_tm",
        else:
            print "broken mapping to parameters"

        theta_offset = np.zeros(theta.shape)
        theta_offset[ind] = step_size
        f1, df1 = model.E_dE(theta+theta_offset)
        df_true = (f1 - f0)/step_size

        print (
            "ind", ind, "df pred", df0[ind], "df true", df_true,
            "df pred - df true", df0[ind] - df_true)


def load_mirt_parameters(tmirt, npz_file):
    """
    Initialize the parameters for the TMIRT using the parameters from the MIRT
    saved in npz_file.
    """

    mirt = np.load(npz_file)

    mirt_theta = mirt["theta"][()]
    mirt_exercise_ind_dict = mirt["exercise_ind_dict"][()]

    assert(mirt_theta.num_abilities == tmirt.num_abilities)

    # step through the exercises in the tmirt
    for ex, idx_tmirt in tmirt.exercise_index.iteritems():
        if ex[1] in mirt_exercise_ind_dict:
            idx_mirt = mirt_exercise_ind_dict[ex[1]]
        else:
            print "missing ", ex[1], " in mirt model"
            continue
        tmirt.W_exercise_correct[idx_tmirt] = mirt_theta.W_correct[idx_mirt]
        tmirt.W_exercise_logtime[idx_tmirt] = mirt_theta.W_time[idx_mirt]
        tmirt.sigma_exercise_logtime[idx_tmirt] = mirt_theta.sigma_time[idx_mirt]


def main():
    options = get_cmd_line_arguments()

    print >>sys.stderr, datetime.datetime.now(), "Starting main.", options

    model = load_data(options)

    # now do num_epochs EM steps
    for epoch in range(options.num_epochs):
        print >>sys.stderr, datetime.datetime.now(), "epoch %d, " % epoch,

        if options.bootstrap_mirt != '' and epoch < options.bootstrap_epochs:
            load_mirt_parameters(model, options.bootstrap_mirt)

        # Expectation step
        # Compute (and print) the energies during learning as a diagnostic.
        # These should decrease.

        E_samples = model.sample_abilities_HMC_natgrad(
            num_steps=options.sampling_num_steps,
            epsilon=options.sampling_epsilon)

        print >>sys.stderr, "E log L %f, " % (
                -np.sum(E_samples) / model.users.num_users / np.log(2.)),

        # debugging info -- accumulate mean and covariance of abilities vector
        mn_a = np.mean(model.users.a, axis=1)
        cov_a = np.mean(model.users.a**2, axis=1)
        print >>sys.stderr, "<abilities>", mn_a,
        print >>sys.stderr, ", <abilities^2>", cov_a, ", ",

        # Maximization step
        old_theta = model.flatten_parameters()
        #L = 0.
        #new_theta = old_theta
        new_theta, L, _ = scipy.optimize.fmin_l_bfgs_b(
            model.E_dE,
            old_theta.copy().ravel(),
            disp=0,
            maxfun=options.max_pass_lbfgs, m=100)
        model.unflatten_parameters(new_theta)

        # Print debugging info on the progress of the training
        print >>sys.stderr, "M log L %f, " % (-L/np.log(2)),
        print >>sys.stderr, "||theta|| %f, " % (
                np.sqrt(np.sum(new_theta ** 2))),
        print >>sys.stderr, "||dtheta|| %f" % (
                np.sqrt(np.sum((new_theta - old_theta) ** 2)))

        # save state as a .npz
        # TODO(jascha) can delete model.users when saving to save space
        np.savez("%s_epoch=%d.npz" % (options.output, epoch),
                 tmirt=model)

        # TODO (jascha) compute log likelihood
        # if np.mod(epoch, 10) == 0 and epoch > 0:
        #     log_likelihood_AIS(
        #         user_states[:options.ais_users], couplings, options)


if __name__ == '__main__':
    main()
