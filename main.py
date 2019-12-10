from copy import deepcopy

import numpy as np
from random import seed
import random

# TODO change learning formula, learning is made each move
seed(1)

ALPHA = 0.7
GAMMA = 0.8  # Discount Factor
STEP_PER_EPOCH = 100
EPOCH = 100
EXPLORATION_CONST = 0
CHANCE_TO_EXPLORE_MAX = 0.9
CHANCE_TO_EXPLORE = CHANCE_TO_EXPLORE_MAX

state_action_dictionary = {}
action_list = []
reward = 0
current_position = (5, 5)
game_map = np.zeros((6, 6))

# Contains look up table for all move possible for all position
# In this order {up, right, down, left}
#   |0|
# |3| |1|
#   |2|
lookup_table = np.arange(6 * 6 * 4).reshape((6, 6, 4))


def tuple_to_key(position_tuple):
    return lookup_table[position_tuple[0], position_tuple[1], position_tuple[2]]


def compute_q(selected_direction, next_position, current_q):
    """
    Q(s,a) = (1- alpha(t))Q(s,a) + alpha(r + alpha * Qmax (S',a)
    Qmax (S',a) : We take max value for state and action for the next action
    :param selected_direction:
    :param next_position:
    :return:
    """
    eq_first_part = (1 - ALPHA) * (
        state_action_dictionary[lookup_table[current_position[0], current_position[1], selected_direction]] if
        lookup_table[current_position[0], current_position[
            1], selected_direction] in state_action_dictionary else EXPLORATION_CONST)

    # Finding Qmax
    q_max = 0.0
    for i in range(3):
        q_max_temp = (
            state_action_dictionary[lookup_table[next_position[0], next_position[1], i]] if
            lookup_table[next_position[0], next_position[
                1], i] in state_action_dictionary else EXPLORATION_CONST)
        if q_max_temp > q_max:
            q_max = q_max_temp

    eq_second_part = ALPHA * (game_map[next_position[0], next_position[1]] + ALPHA * q_max - current_q)
    return eq_first_part + eq_second_part


def set_game_map():
    local_game_map = np.zeros((6, 6))
    # local_game_map = local_game_map - 1
    local_game_map[0, 5] = 1000
    local_game_map[1, 0] = 50
    local_game_map[3, 2] = -10
    local_game_map[3, 3] = -10
    local_game_map[3, 4] = -10
    local_game_map[3, 5] = -10
    local_game_map[4, 2] = -10
    return local_game_map


def q_max_func(previous_state_action_value):
    return GAMMA * previous_state_action_value


def pick_direction(state):
    """
    Random chance to take a random direction to add exploration at the beginning
    Pick a direction based on previous learning, if all the direction are equals in value take a random direction
    :param state:
    :return:
    """
    global state_action_dictionary
    global CHANCE_TO_EXPLORE
    if random.uniform(0, 1) > CHANCE_TO_EXPLORE:
        upper_val = state_action_dictionary[lookup_table[state[0], state[1], 0]] if lookup_table[state[0], state[
            1], 0] in state_action_dictionary else EXPLORATION_CONST
        right_val = state_action_dictionary[lookup_table[state[0], state[1], 1]] if lookup_table[state[0], state[
            1], 1] in state_action_dictionary else EXPLORATION_CONST
        lower_val = state_action_dictionary[lookup_table[state[0], state[1], 2]] if lookup_table[state[0], state[
            1], 2] in state_action_dictionary else EXPLORATION_CONST
        left_val = state_action_dictionary[lookup_table[state[0], state[1], 3]] if lookup_table[state[0], state[
            1], 3] in state_action_dictionary else EXPLORATION_CONST

        if left_val > right_val and left_val > upper_val and left_val > lower_val:
            return 3
        if left_val < right_val and right_val > upper_val and right_val > lower_val:
            return 1
        if upper_val > left_val and right_val < upper_val and upper_val > lower_val:
            return 0
        if lower_val > left_val and right_val < lower_val and lower_val > upper_val:
            return 2
        else:
            return random.randint(0, 3)
    else:
        return random.randint(0, 3)


def convert_direction_to_position(direction):
    if direction == 0:
        return current_position[0] - 1, current_position[1]
    elif direction == 1:
        return current_position[0], current_position[1] + 1
    elif direction == 2:
        return current_position[0] + 1, current_position[1]
    elif direction == 3:
        return current_position[0], current_position[1] - 1


def make_move():
    global state_action_dictionary
    global current_position
    global reward
    global game_map
    global lookup_table
    local_reward = 0.0
    selected_direction = pick_direction(current_position)
    action_list.append(list(current_position) + [selected_direction])
    next_position = 0
    current_position_copy = deepcopy(current_position);
    if current_position[0] == 5 and selected_direction == 2:
        reward = reward - 10
        local_reward = - 10
        next_position = current_position
    elif current_position[1] == 5 and selected_direction == 1:
        reward = reward - 10
        local_reward = - 10
        next_position = current_position
    elif current_position[0] == 0 and selected_direction == 0:
        reward = reward - 10
        local_reward = - 10
        next_position = current_position
    elif current_position[1] == 0 and selected_direction == 3:
        reward = reward - 10
        local_reward = - 10
        next_position = current_position
    else:
        next_position = convert_direction_to_position(selected_direction)
        if game_map[next_position[0], next_position[1]] == -10:
            reward = reward - 10
            local_reward = - 10
            next_position = current_position
        else:
            current_position = next_position
            reward = reward + game_map[current_position[0], current_position[1]]
            local_reward = game_map[current_position[0], current_position[1]]

    # Learning
    # i = lookup_table[current_position[0], current_position[1], [selected_direction]][0]
    i = lookup_table[current_position_copy[0], current_position_copy[1], [selected_direction]][0]

    if i in state_action_dictionary:
        current_q = compute_q(selected_direction, next_position, state_action_dictionary[i])
    else:
        current_q = compute_q(selected_direction, next_position, 0)
    state_action_dictionary[i] = current_q


def update_dict():
    global reward
    global action_list

    current_reward = reward
    copy_action_list = action_list.copy()
    copy_action_list.reverse()
    for action_state in copy_action_list:
        current_reward = current_reward * (1 - ALPHA)
        if tuple_to_key(action_state) in state_action_dictionary:
            state_action_dictionary[tuple_to_key(action_state)] = (
                    state_action_dictionary[tuple_to_key(action_state)] + current_reward)
        else:
            state_action_dictionary[tuple_to_key(action_state)] = current_reward


def draw_world(actual_current_position, current_step):
    copy_map = game_map.copy()
    copy_map[actual_current_position[0], actual_current_position[1]] = 7
    print("current_step = ", current_step)
    print(copy_map)
    print("reward = ", reward)


def print_state_action_dictionary():
    global state_action_dictionary
    global lookup_table
    for key in state_action_dictionary:
        index_in_array = np.where(lookup_table == key)
        print('Key : ', key, '\t', index_in_array[0], index_in_array[1], index_in_array[2], '\t Value : ',
              state_action_dictionary[key])


def run_epoch(number_of_current_epoch):
    """
    Run a single epoch
    Decrease the chance to take random decision at each epoch
    Last epoch should have near 0% chance of taking random decisions
    """
    global CHANCE_TO_EXPLORE
    global action_list
    global reward
    global current_position
    action_list = []
    reward = 0
    current_position = (5, 5)
    for i in range(0, STEP_PER_EPOCH):
        make_move()
        draw_world(current_position, i)
    # update_dict()
    print_state_action_dictionary()
    CHANCE_TO_EXPLORE = CHANCE_TO_EXPLORE_MAX * ((EPOCH - number_of_current_epoch) / EPOCH)
    print('Chance to explore : ', CHANCE_TO_EXPLORE)


def run_sim():
    global game_map
    game_map = set_game_map()
    for i in range(EPOCH):
        print("EPOCH = ", i)
        run_epoch(i)
    print(state_action_dictionary)


run_sim()
