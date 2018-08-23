# -*- coding:utf-8 -*-

import numpy as np

def initialize():
    num_state = 2  #the number of states
    num_action =2  #the number of actions
    q_value = np.zeros((num_state, num_action))   #the Q-value matrix

    trans_pro = np.zeros((num_action, num_state, num_state))   #the transition probability matrix
    trans_rew = np.zeros((num_action, num_state, num_state))   #the transition reward matrix
    trans_time = np.zeros((num_action, num_state, num_state))  # transition time matrix,size:2x2x2
    trans_pro[0, :, :] = np.array([[0.7, 0.3], [0.4, 0.6]])
    trans_pro[1, :, :] = np.array([[0.9, 0.1], [0.2, 0.8]])
    trans_rew[0, :, :] = np.array([[6, -5], [7, 12]])
    trans_rew[1, :, :] = np.array([[10, 17], [-14, 13]])
    trans_time[0, :, :] = np.array([[1, 5], [120, 60]])
    trans_time[1, :, :] = np.array([[50, 75], [7, 2]])

    stat = {"Q_value": q_value, "state": 0, "action": 0, "next_state": 0, "reward": 0, "time": 0,
            "mu": 0.0001}

    return stat, trans_pro, trans_rew, trans_time

def select_action(stat):
    """
    to select an action
    :param stat: type:dict
    :return: action:type:digit
    """
    candidate = 0   #default action
    flag = 0        #a flag that completes to select an action
    rand_num = np.random.rand()

    _, num_actions = stat["Q_value"].shape
    proba = 1/num_actions

    while flag == 0:
        if rand_num < proba:
            flag = 1
        else:
            candidate += 1
            proba += 1/num_actions

    return candidate

def find_next_state(stat, trans_pro):
    """
    to find the next state
    :param stat: type:dict
    :param trans_pro: transition probability matrix
    :return: state
    """
    candidate = 0   #default state
    flag = 0        #a flag that completes to find the next state
    rand_num = np.random.rand()
    proba = trans_pro[stat["action"], stat["state"], candidate]

    while flag == 0:
        if rand_num < proba:
            flag = 1
        else:
            candidate += 1
            proba += trans_pro[stat["action"], stat["state"], candidate]

    return candidate

def alpha_rate(iteration):
    """
    alpha learning rate
    :param iteration: the count of iteration
    :return: alpha
    """
    return 150/(300 + iteration)

def q_learning(stat, iteration):
    """
    to update the Q-value matrix
    :param stat: type:dict
    :param iteration:the count of iteration
    :return: stat
    """
    alpha = alpha_rate(iteration)

    q_value_max = max(stat["Q_value"][stat["next_state"], :])
    q_temp = stat["Q_value"][stat["state"], stat["action"]]
    q_temp = (1 - alpha)*q_temp + alpha*(stat["reward"] + np.exp(-stat["mu"]*stat["time"]))
    stat["Q_value"][stat["state"], stat["action"]] = q_temp

    return stat

def find_policy(stat):
    """
    to find the policy
    :param stat: type:dict
    :return: policy
    """
    num_state = len(stat["Q_value"])
    q_value_func = np.zeros(num_state)
    policy = np.zeros(num_state)

    for iteration in range(num_state):
        q_value_func[iteration] = max(stat["Q_value"][iteration, :])
        policy[iteration] = np.argmax(stat["Q_value"][iteration, :])

    return q_value_func, policy


if __name__ == "__main__":
    IterMax = 10000   #the number of iterations
    stat, trans_pro, trans_rew, trans_time = initialize()
    #print("the information of the stat is:\n", stat)
    #print("the transition probability matrix is:\n", trans_pro)
    #print("the transition reward matrix is:\n", trans_rew)

    for iteration in range(IterMax):
        action = select_action(stat)
        stat["action"] = action
        next_state = find_next_state(stat, trans_pro)
        stat["next_state"] = next_state
        stat["reward"] = trans_rew[action, stat["state"], next_state]
        stat["time"] = trans_time[action, stat["state"], next_state]

        stat = q_learning(stat, iteration)
        stat["state"] = next_state

    q_value_func, policy = find_policy(stat)
    print("The policy is: ", policy + 1)
    print("The Q-value function is:", q_value_func)
    print("The Q-value matrix is:\n", stat["Q_value"])