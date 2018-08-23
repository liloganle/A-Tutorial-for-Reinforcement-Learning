# -*- coding:utf-8 -*-

import numpy as np

def initialize():
    """
    to initialize the parameters
    :return: stat:dict, trans_pro:matrix, trans_rew:matrix
    """
    num_state = 2    #the number of states
    num_action = 2   #the number of actions
    q_value = np.zeros((num_state, num_action))  #the Q-value matrix

    trans_pro = np.zeros((num_action, num_state, num_state))  #transition probability matrix, size:2x2x2
    trans_rew = np.zeros((num_action, num_state, num_state))  #transition reward matrix, size:2x2x2

    trans_pro[0, :, :] = np.array([[0.7, 0.3], [0.4, 0.6]])
    trans_pro[1, :, :] = np.array([[0.9, 0.1], [0.2, 0.8]])
    trans_rew[0, :, :] = np.array([[6, -5], [7, 12]])
    trans_rew[1, :, :] = np.array([[10, 17], [-14, 13]])
    stat = {"Q_value": q_value, "state": 0, "action": 0, "next_state": 0, "reward": 0, "gama": 0.8}

    return stat, trans_pro, trans_rew

def select_action(stat):
    """
    to select an action
    :param stat: type:dict
    :return: candidate:a action
    """
    rand_num = np.random.rand()  #to generate a random number between 0 and 1
    _, num_actions = stat["Q_value"].shape
    proba = 1/num_actions
    candidate = 0  #candidate action
    flag = 0  # a flag that completes to select a action

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
    :param trans_pro: type:matrix
    :return: candidate:a state
    """
    candidate = 0  #candidate state
    flag = 0  #a flag that completes to find a state
    rand_num = np.random.rand()  #to generate a random number between 0 and 1
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
    the learning rate: alpha
    :param iteration: the count of iteration
    :return: alpha learning rate
    """
    #return np.log(iteration+1)/(iteration+1)
    return 150/(300+iteration)


def q_learning(stat, iteration):
    """
    update the parameter:stat
    :param stat: dict
    :param iteration: the count of iteration
    :return: stat
    """
    alpha = alpha_rate(iteration)

    q_value_max = max(stat["Q_value"][stat["next_state"], :])
    q_temp = stat["Q_value"][stat["state"], stat["action"]]
    q_temp = (1 - alpha)*q_temp + alpha*(stat["reward"] + stat["gama"]*q_value_max)
    stat["Q_value"][stat["state"], stat["action"]] = q_temp

    return stat

def find_policy(stat):
    """
    to find the policy
    :param stat:
    :return: policy
    """
    num_state = len(stat["Q_value"])
    policy = np.zeros(num_state)
    q_value_func = np.zeros(num_state)

    for state in range(num_state):
        q_value_func[state] = max(stat["Q_value"][state, :])
        policy[state] = np.argmax(stat["Q_value"][state, :])

    return q_value_func, policy


if __name__ == "__main__":
    IterMax = 10000  #the number of iterations
    stat, trans_pro, trans_rew = initialize()
    #print("the initialized state of the system is:\n", stat)
    #print("the transition probability matrix is:\n", trans_pro)
    #print("the transition reward matrix is:\n", trans_rew)

    for iteration in range(IterMax):
        action = select_action(stat)
        stat["action"] = action
        next_state = find_next_state(stat, trans_pro)
        stat["next_state"] = next_state
        stat["reward"] = trans_rew[action, stat["state"], next_state]

        stat = q_learning(stat, iteration + 1)
        stat["state"] = next_state

    q_value_func, policy = find_policy(stat)

    print("the policy is: ", policy + 1)
    print("the Q-value function is: ", q_value_func)
    print("the Q-value is:\n", stat["Q_value"])
