import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_data(fname):
    a = np.load(fname)
    return a['tmirt'][()]


def plot_learning_trajectories(tmirt, num_trajectories=25, min_length=10, use_time=False):
    plt.figure()
    plt.clf()

    num_plotted = 0

    while num_plotted < num_trajectories:
        user_i = np.random.randint(tmirt.users.num_users)
        idx = np.nonzero((tmirt.users.a_to_user.reshape((-1)) == user_i))[0]
        if idx.shape[0] > min_length:
            num_plotted += 1
            y = tmirt.users.a[0, idx]
            plt.ylabel('a[0]')
            if tmirt.num_abilities > 1 and not use_time:
                x = tmirt.users.a[1, idx]
                plt.xlabel('a[1]')
            else:
                x = np.arange(idx.shape[0])
                plt.xlabel('resource #')

            plt.plot(x, y, '+')
            plt.plot(x[[0, -1]], y[[0, -1]], '-')
            plt.plot(x[0], y[0], 'go')
            plt.plot(x[-1], y[-1], 'yo')

    plt.show()


def plot_predicted_accuracy_vs_time(tmirt, num_trajectories=25, min_length=10):
    """
    plot the average accuracy on all exercises vs. time for num_trajectories
    students with at least min_length terms in their histories
    """
    plt.figure()
    plt.clf()
    plt.ylabel('Average predicted accuracy')
    plt.xlabel('Interactions')

    num_plotted = 0

    W = tmirt.W_exercise_correct

    mean_pred_accuracy = 0.    
    while num_plotted < num_trajectories:
        user_i = np.random.randint(tmirt.users.num_users)
        idx = np.nonzero((tmirt.users.a_to_user.reshape((-1)) == user_i))[0]
        if idx.shape[0] > min_length:
            num_plotted += 1

            pred_accuracy = np.zeros((idx.shape[0]))
            for tt in range(idx.shape[0]):
                a = tmirt.users.a[:, [idx[tt]]]
                a = np.vstack((a, np.ones((1, 1))))
                pred_accuracy[tt] = np.mean(sigmoid(np.dot(W, a)))
            mean_pred_accuracy += pred_accuracy[:min_length]/num_trajectories

            plt.plot(np.arange(idx.shape[0]), pred_accuracy, ':')

    plt.plot(np.arange(min_length), mean_pred_accuracy, linewidth=4)
    plt.show()


def sigmoid(u):
    return 1. / (1. + np.exp(-u))


def get_chance_abilities(tmirt, exercise):
    """Return the smallest abilities vector that predicts 50/50 accuracy
    on 'exercise'."""
    Wfull = tmirt.W_exercise_correct[tmirt.exercise_index[exercise], :]

    # peel off bias
    b = Wfull[-1]
    W = Wfull[:-1]

    # find the smallest abilities for which prob. of correct
    # is 50%
    a = -W*b / np.sum(W**2)
    a = a.reshape((-1, 1))
    # add bias unit
    a = np.vstack((a, np.ones((1, 1))))

    return a


def get_post_resource_prediction(tmirt, ex_nms, rcs):
    # matrix to hold predicted correctness after resource application
    PC = np.ones((len(rcs), len(ex_nms)))*0.5
    # populate the matrix
    for i_ex, ex in enumerate(ex_nms):
        for i_rc, rc in enumerate(rcs):
            a = get_chance_abilities(tmirt, ex)

            # predict the abilities after resource exposure
            try:
                Phi = tmirt.Phi[:, :, tmirt.resource_index[rc]]
            except:
                # TODO(jace) : what exception is this catching? Was this
                # due to a change/inconsistency in model storage format?
                rc = (rc[0], rc[1], (rc[2] == 1))  # convert to bool
                try:
                    Phi = tmirt.Phi[:, :, tmirt.resource_index[rc]]
                except:
                    continue
            a_pred = a[:-1, :] + np.dot(Phi, a)
            # add bias unit
            a_pred = np.vstack((a_pred, np.ones((1, 1))))

            Wfull = tmirt.W_exercise_correct[tmirt.exercise_index[ex], :]
            PC[i_rc, i_ex] = sigmoid(np.dot(Wfull, a_pred))

    return PC


def exercises_sorted_by_difficulty(tmirt):
    """Returns tuples of the from ("exercise", ex_name), sorted by
    increasing difficulty."""
    exercise_ind_dict = tmirt.exercise_index
    couplings = tmirt.W_exercise_correct
    tt = [(
          couplings[exercise_ind_dict[nm], -1], nm)
          for nm in exercise_ind_dict.keys()]
    tt = sorted(tt, key=lambda tl: -tl[0])
    exercises_sorted = [tl[1] for tl in tt]
    return exercises_sorted


def exercise_resource_keys(exercises, correct):
    # concatenate correct (a constant, 1 or 0) into each tuple in exercises
    return [e + (correct,) for e in exercises]


def compute_resource_exercise_improvement(tmirt):
    """Compute the effect on exercise accuracy of exposure to resources."""

    exs_sorted = exercises_sorted_by_difficulty(tmirt)

    # resources from exercise, correct
    rcs = exercise_resource_keys(exs_sorted, 0)
    PC_ex0 = get_post_resource_prediction(tmirt, exs_sorted, rcs)

    rcs = exercise_resource_keys(exs_sorted, 1)
    PC_ex1 = get_post_resource_prediction(tmirt, exs_sorted, rcs)

    rcs = []
    for key in tmirt.resource_index:
        if key[0] == 'video':
            rcs.append(key)
    if len(rcs) > 0:
        PC_vid = get_post_resource_prediction(tmirt, exs_sorted, rcs)
    else:
        PC_vid = np.zeros((10, 10))

    return exs_sorted, PC_ex0, PC_ex1, PC_vid


def plot_resource_exercise_improvement(tmirt):
    """ plot the predicted effect of a resource on exercise performance
    assuming initial state is chance performance """

    ex_nms, PC_ex0, PC_ex1, PC_vid = (
            compute_resource_exercise_improvement(tmirt))

    plt.figure()
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.title('Effect of INCORRECT answers')
    plt.imshow(PC_ex0, interpolation='nearest', cmap=matplotlib.cm.Greys_r)
    plt.xlabel('Target Exercise (increasing difficulty)')
    plt.ylabel('Resource Exercise (increasing difficulty)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Effect of CORRECT answers')
    plt.imshow(PC_ex1, interpolation='nearest', cmap=matplotlib.cm.Greys_r)
    plt.xlabel('Target Exercise (increasing difficulty)')
    plt.ylabel('Resource Exercise (increasing difficulty)')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Effect of VIDEOS')
    plt.imshow(PC_vid, interpolation='nearest', cmap=matplotlib.cm.Greys_r)
    plt.xlabel('Target Exercise (increasing difficulty)')
    plt.ylabel('Resource Video')
    plt.colorbar()

    plt.show()


def compute_cross_resource_prediction(tmirt, exercises):
    """Compute a matrix of predicted accuracy on 'rcs' resources, given 50/50
    accuracy on 'ex_nms' exercises."""

    # initialize results matrix
    RC = np.zeros((len(exercises), len(exercises)))

    # populate the matrix
    for i_ex, ex in enumerate(exercises):
        for i_ex2, ex2 in enumerate(exercises):
            a = get_chance_abilities(tmirt, ex)

            W_ex2 = tmirt.W_exercise_correct[tmirt.exercise_index[ex2], :]

            RC[i_ex2, i_ex] = sigmoid(np.dot(W_ex2, a))

    return RC


def print_top_suggestions_per_exercise(tmirt):
    exs_sorted = exercises_sorted_by_difficulty(tmirt)
    exs_short = [e[1] for e in exs_sorted]

    n = len(exs_sorted)
    fifty_fifty = np.ones((n, n)) * .5

    # for 50% accuracy in the column index exercise, holds the predicted
    # accuracy in the row index exercise.
    cross_accuracy = compute_cross_resource_prediction(tmirt, exs_sorted)

    # for each exercise (column), compute the expected gain or loss in accuracy
    # conditional on getting another problem in each exercise (row) right/wrong
    rcs = exercise_resource_keys(exs_sorted, 0)
    cond_accuracy_0 = get_post_resource_prediction(tmirt, exs_sorted, rcs)
    cond_gain_0 = cond_accuracy_0 - fifty_fifty

    rcs = exercise_resource_keys(exs_sorted, 1)
    cond_accuracy_1 = get_post_resource_prediction(tmirt, exs_sorted, rcs)
    cond_gain_1 = cond_accuracy_1 - fifty_fifty

    expected_gain = cross_accuracy * cond_gain_1 + (
            (1 - cross_accuracy) * cond_gain_0)

    top_suggestions = []
    for i, ex in enumerate(exs_short):
        row_output = "%s" % ex
        num_suggestions_per_ex = 3
        for j in range(num_suggestions_per_ex):
            suggestion_index = np.argmax(expected_gain[:, i])
            row_output += ", %s" % exs_short[suggestion_index]
            # Overwite the max gain with something small, so the next
            # iteration will select a new argmax.
            expected_gain[suggestion_index, i] = -1.0

        print row_output


def generate_csv_files(tmirt, fname=None):
    """ these should all have negative real parts! """

    if fname is None:
        fname = "abilities=%d_exercises=%d_resources=%d_users=%d" % (
                tmirt.num_abilities, tmirt.num_exercises,
                tmirt.num_resources, tmirt.users.num_users)

    ## resource eigenvalue view
    f1 = open(fname + '_resource_eig.csv', 'w+')
    nms = sorted(tmirt.resource_index.keys())
    for ii in range(tmirt.Phi.shape[0]):
        print >>f1, "bias%d," % (ii),
    for ii in range(tmirt.Phi.shape[0]):
        print >>f1, "weight_eig%d," % (ii),
    for ii in range(tmirt.Phi.shape[0]):
        print >>f1, "J_eig%d," % (ii),
    print >>f1, "resource name"

    for nm in nms:
        Phi = tmirt.Phi[:, :, tmirt.resource_index[nm]]
        J = tmirt.J[:, :, tmirt.resource_index[nm]]
        for ii in range(tmirt.Phi.shape[0]):
            print >>f1, Phi[ii, -1], ',',
        eg = np.linalg.eig(Phi[:, :-1])[0]
        for ii in range(tmirt.Phi.shape[0]):
            print >>f1, eg[ii], ',',
        eg = np.linalg.eig(J)[0]
        for ii in range(tmirt.Phi.shape[0]):
            print >>f1, eg[ii], ',',
        print >>f1, str(nm)
    f1.close()

    ## exercise correctness
    f1 = open(fname + "_exercise_correct.csv", 'w+')
    couplings = tmirt.W_exercise_correct

    exercise_ind_dict = tmirt.exercise_index
    tt = [(couplings[exercise_ind_dict[nm], :-1],
           couplings[exercise_ind_dict[nm], -1], nm)
          for nm in exercise_ind_dict.keys()]
    tt = sorted(tt, key=lambda tl: tl[1])
    print >>f1, 'bias, ',
    for ii in range(tt[0][0].shape[0]):
        print >>f1, "coupling %d, " % ii,
    print >>f1, 'exercise name'
    for t in tt:
        print >>f1, t[1], ',',
        for ii in range(tt[0][0].shape[0]):
            print >>f1, t[0][ii], ',',
        print >>f1, t[2]
    f1.close()

    ## abilities update parameters
    f1 = open(fname + "_resource.csv", 'w+')
    nms = sorted(tmirt.resource_index.keys())
    for ii in range(tmirt.Phi.shape[0]):
        print >>f1, "bias%d," % (ii),
    for ii in range(tmirt.Phi.shape[0]):
        for jj in range(tmirt.Phi.shape[1]-1):
            print >>f1, "weight%d_%d," % (ii, jj),
    for ii in range(tmirt.J.shape[0]):
        for jj in range(tmirt.J.shape[1]):
            print >>f1, "J%d_%d," % (ii, jj),
    print >>f1, "resource name"

    for nm in nms:
        Phi = tmirt.Phi[:, :, tmirt.resource_index[nm]]
        J = tmirt.J[:, :, tmirt.resource_index[nm]]
        for ii in range(tmirt.Phi.shape[0]):
            print >>f1, Phi[ii, -1], ',',
        for ii in range(tmirt.Phi.shape[0]):
            for jj in range(tmirt.Phi.shape[1]-1):
                print >>f1, Phi[ii, jj], ',',
        for ii in range(tmirt.J.shape[0]):
            for jj in range(tmirt.J.shape[1]):
                print >>f1, J[ii, jj], ',',
        print >>f1, str(nm)
    f1.close()


plt.ion()
#fname = 'fracs_godot__epoch=425.npz'
#fname = 'data/tmirt_file=all_sorted_n_filtered_abilities=2_epoch=10.npz'
#fname = 'data/tmirt_file=all_sorted_n_filtered_abilities=10_epoch=3.npz'
#fname = 'data/tmirt_file=user_assessment.responses_abilities=2_epoch=222.npz'


def analyze(fname):
    tmirt = load_data(fname)
    generate_csv_files(tmirt)
    plot_learning_trajectories(tmirt)
    plot_predicted_accuracy_vs_time(tmirt)
    plot_resource_exercise_improvement(tmirt)
    print_top_suggestions_per_exercise(tmirt)
