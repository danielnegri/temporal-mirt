import argparse
import accuracy_model_util as acc_util
import sys
from temporal_mirt import TMIRTResource
import fileinput
import numpy as np
linesplit = acc_util.linesplit


def load_data_to_test(model, indexer='plog', fname='/data/eliana/first_problem_log/test', max_users=np.inf):
    """
    Create a new TMIRT model, and load it with training data.
    """

    prev_user = None
    resources = []

    idx_pl = acc_util.get_FieldIndexer(indexer)

    last_answers = []

    print >>sys.stderr, "loading data"
    # loop through all the training data, and create user objects
    for line in fileinput.input(fname):
        if model.users.num_users == max_users:
            break
        # split on either tab or \x01 so the code works via Hive or pipe
        row = linesplit.split(line.strip())
        # the user and timestamp are shared by all row types.
        # load the user
        user = row[idx_pl.user]
        if user != prev_user:
            # We're getting a new user, so perform the reduce operation
            # on our previous user
            if len(resources) > 1:
                model.users.add_user(user, resources)
                last_answers.append(resources[-1])
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

    if len(resources) > 1 and model.users.num_users < max_users:
        # flush the data for the final user, too
        model.users.add_user(user, resources[:-1])
        last_answers.append(resources[-1])

    fileinput.close()
    # create parameter structures, change datatypes, etc
    model.finalize_users()
    return last_answers


def get_cmd_line_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with predicted and actual accuracy "
                    "on test data, and an NPZ file holding the resulting "
                    "model state.")
    parser.add_argument("model", type=str, help="TMIRT model filename.")
    parser.add_argument("test", type=str, help="Test dataset filename.")
    parser.add_argument("out", type=str, default="test_predictions", help="Base filename for output.")
    parser.add_argument("-m", "--max_users", type=int, default=1e4,
        help="The maximum number of users to run prediction on.")
    parser.add_argument("-n", "--num_samples", type=int, default=20,
        help="The number of samples to average over for the prediction.")
    parser.add_argument("-s", "--hmc_steps", type=int, default=10,
        help="The number of sampling steps to take between samples.  Burnin will be 10 times this number.")
    parser.add_argument("-L", "--LBFGS_steps", type=int, default=100,
        help="The number of LBFGS steps to take for the MAP estimate.")

    # DEBUG use parse_known_args rather than parse_args so can easily run it
    # inside pylab
    options, _ = parser.parse_known_args()

    return options


def main():
    options = get_cmd_line_arguments()

    print options

    tmirt = np.load(options.model)
    tmirt = tmirt['tmirt'][()]
    tmirt.reset_users()
    last_answers = load_data_to_test(tmirt, max_users=options.max_users)
    print "Number users %d"%(tmirt.users.num_users)
    assert(tmirt.users.num_users == len(last_answers))
    tmirt.predict_performance(N_avg_samp=options.num_samples, hmc_steps=options.hmc_steps, hmc_burnin=options.hmc_steps*10)

    f = open(options.out + ".roc", 'w')
    for i in range(len(last_answers)):
        f.write(
            str(1 if last_answers[i].correct else 0) + ' ' +
            str(tmirt.predict_user_exercise_performance(
                i, last_answers[i].name))+'\n')

    np.savez(options.out + '.npz', model=tmirt, predict=tmirt.predict_performance(),
             last=last_answers)

if __name__ == '__main__':
    main()
