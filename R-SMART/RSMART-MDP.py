# -*- coding:utf-8 -*-

import numpy as np

def initialize():
    """
    to initialize the parameters
    :return: stat: type dict, trans_pro: transition probability, trans_rew: transition reward
    """
    num_state = 2   #the number of states
    num_action = 2  #the number of actions
    q_value = np.zeros((num_state, num_action))  #size:2x2

    trans_pro = np.zeros((num_action, num_state, num_state))  #transition probability matrix,size:2x2x2
    trans_rew = np.zeros((num_action, num_state, num_state))  #transition reward matrix,size:2x2x2
    trans_pro[0, :, :] = np.array([[0.7, 0.3], [0.4, 0.6]])
    trans_pro[1, :, :] = np.array([[0.9, 0.1], [0.2, 0.8]])
    trans_rew[0, :, :] = np.array([[6, -5], [7, 12]])
    trans_rew[1, :, :] = np.array([[10, 17], [-14, 13]])

    rho = 0.001  # the average reward


    stat = {"Q_value": q_value, "state": 0, "action": 0, "next_state": 0, "reward": 0,
            "total_reward": 0, "rho": rho, "flag": 0, "explore": 0.5}

    return stat, trans_pro, trans_rew

def select_action(stat):
    """
    select action of the next state
    :param stat: type dict
    :return: stat, the next action
    """
    rand_num = np.random.rand()
    if rand_num < (1-stat["explore"]):
        action = np.argmax(stat["Q_value"][stat["state"], :])
    else:
        action = np.argmin(stat["Q_value"][stat["state"], :])

    if stat["Q_value"][stat["state"], action] == max(stat["Q_value"][stat["state"], :]):
        stat["flag"] = 0
    else:
        stat["flag"] = 1

    stat["explore"] = stat["explore"] * 0.999

    return stat, action

def find_next_state(stat, trans_pro):
    """
    this function is going to find the next state with current state and action
    :param stat: type dict
    :param trans_pro: transition probability matrix
    :return: the next state
    """
    candidate = 0
    complete = 0  #the flag that completes to find the next state
    proba = trans_pro[stat["action"], stat["state"], candidate]

    rand_number = np.random.rand()
    #print("the rand number is: ", rand_number)

    while complete == 0:
        if rand_number < proba:
            complete = 1
        else:
            candidate += 1
            proba += trans_pro[stat["action"], stat["state"], candidate]

    return candidate

#next_state = find_next_state(state, action, trans_pro)
#print("the next state is: ", next_state)
#reward = trans_rew[action, state, next_state]

def alpha_rate(iteration):
    """
    learning rate: alpha_k
    :param iteration
    :return: alpha
    """
    return np.log(iteration+1)/(iteration+1)

def beta_rate(iteration):
    """
    learning rate: beta_k
    :param iteration:
    :return: beta
    """
    return 9/(100+iteration)

def update(stat, iteration):
    """
    update the information of Q-factor
    :param stat: type dict
    :param iteration: the number of iteration
    :return: stat
    """
    q_value_max = max(stat["Q_value"][stat["next_state"], :])
    alpha = alpha_rate(iteration)
    q_temp = stat["Q_value"][stat["state"], stat["action"]]
    q_temp = (1 - alpha) * q_temp + alpha * (stat["reward"] - stat["rho"] + 0.99 * q_value_max)

    stat["Q_value"][stat["state"], stat["action"]] = q_temp

    if stat["flag"] == 0:
        beta = beta_rate(iteration)
        stat["total_reward"] += stat["reward"]
        stat["rho"] = (1-beta)*stat["rho"] + beta * stat["total_reward"]/(iteration+1)

    return stat


def find_policy(stat):
    """
    to feedback the policy
    :param stat:type dict
    :return: q_value_function, policy
    """
    num_state = len(stat["Q_value"])
    policy = np.zeros(num_state)
    q_value_function = np.zeros(num_state)
    for iter in range(num_state):
        q_value_function[iter] = max(stat["Q_value"][iter, :])
        policy[iter] = np.argmax(stat["Q_value"][iter, :])

    return q_value_function, policy


if __name__=="__main__":
    IterMax = 10000  # the number of iterations
    stat, trans_pro, trans_rew = initialize()   #initialize the parameters
    #print("the information of the state is:\n", stat)
    #print("the transition probability matrix is:\n", trans_pro)
    #print("the transition reward matrix is:\n", trans_rew)

    for iteration in range(IterMax):
        state = stat["state"]
        stat, action = select_action(stat)  #select action from the initialized state
        stat["action"] = action
        next_state = find_next_state(stat, trans_pro)  #according to initialized state and action
                                                       # to find next state

        stat["next_state"] = next_state
        stat["reward"] = trans_rew[action, state, next_state]

        stat = update(stat, iteration)
        #stat, next_action = select_action(stat)

        stat["state"] = next_state
        #stat["action"] = next_action

    q_value_func, policy = find_policy(stat)

    print("the 10000-th rho is: ", stat["rho"])
    print("the rho is: ", stat["total_reward"]/IterMax)
    print("the Q-value is: \n", stat["Q_value"])
    print("the value function is: ", q_value_func)
    print("the policy is: ", policy+1)