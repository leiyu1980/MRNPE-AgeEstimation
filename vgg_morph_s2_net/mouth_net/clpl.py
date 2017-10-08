#!/usr/bin/python
__author__ = 'palmchou'
import sys
import re
from matplotlib import pyplot as plt
import numpy as np
do_plot = True
if len(sys.argv) < 2:
    print "Usage: plot_log log_file"
    exit(1)
    log_path = '.' # disable IDE warning
else:
    log_path = sys.argv[1]
if len(sys.argv) == 3 and sys.argv[2] == '-np':
    do_plot = False

log_file = open(log_path, 'r')
train_pattern = re.compile('Iteration (\d*), loss = ([-+]?[0-9]*\.?[0-9]+)')
# I0309 15:59:47.188685 21484 solver.cpp:337] Iteration 11000, Testing net (#0)
test_i_pattern = re.compile('Iteration (\d+), Testing net')
# :404]     Test net output #0: loss = 1.13276 (* 1 = 1.13276 loss)
test_l_pattern = re.compile('Test net output #11: loss = ([-+]?[0-9]*\.?[0-9]+) \(')
# Test net output #0: accuracy = 0.66875
test_acc_pattern = re.compile('Test net output #0: accuracy = ([-+]?[0-9]*\.?[0-9]+)')
test_i = []
test_l = []
test_acc = []
train_i = []
train_loss = []


loss_max = 0
loss_max_sec = 0
highest_acc = 0
for line in log_file:
    t = test_i_pattern.search(line)
    if t:
        i = int(t.group(1))
        test_i.append(i)
        continue
    tl = test_l_pattern.search(line)
    if tl:
        l = float(tl.group(1))
        test_l.append(l)
        continue
    tacc = test_acc_pattern.search(line)
    if tacc:
        acc = float(tacc.group(1))
        test_acc.append(acc)
        if acc > highest_acc:
            highest_acc = acc
        continue
    g = train_pattern.search(line)
    if g:
        # print g.group(1), g.group(2)
        i = int(g.group(1))
        loss = float(g.group(2))
        if loss > loss_max:
            loss_max = loss
        if loss_max_sec < loss < loss_max:
            loss_max_sec = loss
        train_i.append(i)
        train_loss.append(loss)
        continue
if len(test_i) != len(test_l):
    del test_i[-1]
if len(test_i) <= 2:
    print "Can't find enough loss value! Please check your log file."
    exit(2)
if len(test_i) != len(test_acc):
    del test_i[-1]


for i in range(0, len(test_i)):
    if len(test_acc) == 0:
        print "Test %d loss: %f" % (test_i[i], test_l[i])
    else:
       # print test_i[i]
       # print test_l[i]
       # print test_acc[i]
        print "Test %7d loss: %2.6f acc: %2.6f" % (test_i[i], test_l[i], test_acc[i])

test_i = test_i[1:]
test_l = test_l[1:]
test_acc = test_acc[1:]

if len(test_acc) > 0:
    idx = test_acc.index(highest_acc)
    iter_ = test_i[idx]
    print "Highest accuracy is %f at iteration %d" % (highest_acc, iter_)
if do_plot:
    test_loss, = plt.plot(test_i, test_l, 'r', label='test loss')
    train_loss, = plt.plot(np.array(train_i), np.array(train_loss), 'b', label='train loss')
    accuracy, = plt.plot(test_i, test_acc, 'g', label='accuracy')
    plt.legend()
    plt.ylim([0, max(loss_max_sec, highest_acc)*1.1])
    plt.show()
