import numpy as np
from temporal_mirt import TMIRT, TMIRTResource
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
        user_i = np.random.randint(tmirt.num_users)
        idx = np.nonzero((tmirt.a_to_user.reshape((-1)) == user_i))[0]
        if idx.shape[0] > min_length:
            y = tmirt.a[0, idx]
            plt.ylabel('a[0]')
            if tmirt.num_abilities > 1 and not use_time:
                x = tmirt.a[1, idx]
                plt.xlabel('a[1]')
            else:
                x = np.arange(idx.shape[0])
                plt.xlabel('resource #')

            plt.plot(x, y, '+')
            plt.plot(x[[0,-1]], y[[0,-1]], '-')
            plt.plot(x[0], y[0], 'go')
            plt.plot(x[-1], y[-1], 'yo')
            num_plotted += 1

    plt.show()


def sigmoid(u):
    return 1. / (1. + np.exp(-u))


def get_post_resource_prediction(tmirt, ex_nms, rcs):
    # matrix to hold predicted correctness after resource application
    PC = np.ones((len(rcs), len(ex_nms)))*0.5
    # populate the matrix
    for i_ex, ex in enumerate(ex_nms):
        for i_rc, rc in enumerate(rcs):
            Wfull = tmirt.W_exercise_correct[tmirt.exercise_index[ex],:]
            # peel off bias
            b = Wfull[-1]
            W = Wfull[:-1]

            # find the smallest abilities for which prob. of correct
            # is 50%
            a = -W*b / np.sum(W**2)
            a = a.reshape((-1,1))
            # add bias unit
            a = np.vstack((a,np.ones((1,1))))

            # predict the abilities after resource exposure
            try:
                Phi = tmirt.Phi[:, :, tmirt.resource_index[rc]]
            except:
                rc = (rc[0], rc[1], (rc[2] == 1)) # convert to bool
                try:
                    Phi = tmirt.Phi[:, :, tmirt.resource_index[rc]]
                except:
                    continue
            a_pred = a[:-1,:] + np.dot(Phi, a)
            # add bias unit
            a_pred = np.vstack((a_pred,np.ones((1,1))))
            
            PC[i_rc, i_ex] = sigmoid(np.dot(Wfull, a_pred))

    return PC


def plot_resource_exercise_improvement(tmirt):
    """ plot the predicted effect of a resource on exercise performance
    assuming initial state is chance performance """

    # exercises sorted by increasing difficulty
    exercise_ind_dict = tmirt.exercise_index
    couplings = tmirt.W_exercise_correct
    tt = [(
          couplings[exercise_ind_dict[nm], -1], nm)
          for nm in exercise_ind_dict.keys()]
    tt = sorted(tt, key=lambda tl: -tl[0])
    ex_nms = [tl[1] for tl in tt]

    # resources from exercise, correct
    rcs = []
    crct = 0
    for nm in ex_nms:
        key = (nm[0], nm[1], crct)
        rcs.append(key)
    PC_ex0 = get_post_resource_prediction(tmirt, ex_nms, rcs)

    rcs = []
    crct = 1
    for nm in ex_nms:
        key = (nm[0], nm[1], crct)
        rcs.append(key)
    PC_ex1 = get_post_resource_prediction(tmirt, ex_nms, rcs)

    rcs = []
    for key in tmirt.resource_index:
        if key[0] == 'video':
            rcs.append(key)
    if len(rcs)>0:
        PC_vid = get_post_resource_prediction(tmirt, ex_nms, rcs)
    else:
        PC_vid = np.zeros((10,10))

    # TODO(jascha) add other kinds of resources here

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


def generate_csv_files(tmirt, fname=None):
    """ these should all have negative real parts! """


    if fname == None:
        fname = "abilities=%d_exercises=%d_resources=%d_users=%d"%(tmirt.num_abilities, tmirt.num_exercises, tmirt.num_resources, tmirt.num_users)


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
    plot_resource_exercise_improvement(tmirt)
