import accuracy_model_util as acc_util
import sys
from temporal_mirt import TMIRTResource
import fileinput
import numpy as np
linesplit = acc_util.linesplit


def load_data_to_test(model, indexer='plog', fname='data/test'):
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
        # split on either tab or \x01 so the code works via Hive or pipe
        row = linesplit.split(line.strip())
        # the user and timestamp are shared by all row types.
        # load the user
        user = row[idx_pl.user]
        if user != prev_user and len(resources) > 1:
            # We're getting a new user, so perform the reduce operation
            # on our previous user
            model.users.add_user(user, resources[:-1])
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

    if len(resources) > 1:
        # flush the data for the final user, too
        model.users.add_user(user, resources)
        last_answers.append(resources[-1])
        model.users.add_user(user, resources[:-1])

    fileinput.close()
    # create parameter structures, change datatypes, etc
    model.finalize_users()
    return last_answers


tmirt = np.load("data/tmirt_file=train_abilities=3_epoch=17.npz")
tmirt = tmirt['tmirt'][()]
tmirt.reset_users()
last_answers = load_data_to_test(tmirt)
tmirt.sample_abilities_HMC_natgrad()

f = open('roc', 'w')
for i in range(len(last_answers)):
    f.write(
        str(last_answers[i].correct) + ',' +
        str(tmirt.predict_user_exercise_performance(
            i, last_answers[i].name))+'\n')

np.savez('out.npz', model=tmirt, predict=tmirt.predict_performance(),
         last=last_answers)
