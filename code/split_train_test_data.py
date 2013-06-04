import fileinput
import accuracy_model_util as acc_util
# used to index the fields in with a line of text in the input data file
linesplit = acc_util.linesplit

last_user = None
num_users = 0
test_every = 5

test = open('test', 'w')
train = open('train', 'w')

lines_current_user = []
for line in fileinput.input():
    # split on either tab or \x01 so the code works via Hive or pipe
    row = linesplit.split(line.strip())
    user = row[0]
    if user != last_user:
        num_users += 1
        if num_users % test_every == 0:
            for l in lines_current_user:
                test.write(l)
        else:
            for l in lines_current_user:
                train.write(l)
        current = []
    lines_current_user.append(line)

if num_users % test_every == 0:
    for l in lines_current_user:
        test.write(l)
else:
    for l in lines_current_user:
        train.write(l)
