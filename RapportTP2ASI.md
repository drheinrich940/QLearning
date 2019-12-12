# Rapport TP2 ASI

## Architecture du programme

## Epoch

```python
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


```

## Mouvement

Un mouvement est effectué à la fois, le la **Q table** (ici `state_action_dictionary`) est updaté à chaque fois.

Les mouvements *illégaux* comme se déplacer dans un mur sont évalués mais ne sont pas réalisés.

```python
def make_move():
    global state_action_dictionary
    global current_position
    global reward
    global game_map
    global lookup_table
    selected_direction = pick_direction(current_position)
    action_list.append(list(current_position) + [selected_direction])
    current_position_copy = deepcopy(current_position);
    if current_position[0] == 5 and selected_direction == 2:
        reward = reward - 10
        next_position = current_position
    elif current_position[1] == 5 and selected_direction == 1:
        reward = reward - 10
        next_position = current_position
    elif current_position[0] == 0 and selected_direction == 0:
        reward = reward - 10
        next_position = current_position
    elif current_position[1] == 0 and selected_direction == 3:
        reward = reward - 10
        next_position = current_position
    else:
        next_position = convert_direction_to_position(selected_direction)
        if game_map[next_position[0], next_position[1]] == -10:
            reward = reward - 10
            next_position = current_position
        else:
            current_position = next_position
            reward = reward + game_map[current_position[0], current_position[1]]

    # Learning
    i = lookup_table[current_position_copy[0], current_position_copy[1], [selected_direction]][0]
    if i in state_action_dictionary:
        current_q = compute_q(selected_direction, next_position, state_action_dictionary[i])
    else:
        current_q = compute_q(selected_direction, next_position, 0)
    state_action_dictionary[i] = current_q
```

### Choix de la direction

La direction à prendre à une chance (`CHANCE_TO_EXPLORE`) d'etre choisi aux hasard pour encourager l'exploration et réduire la possibilité de rester coincé sur un maximum local.

Cette chance décroit au fur du temps selont cette formule : $p = 0.9 * \frac{n-i}{n}$ 

- $n$ : Nombre d'epoch à réaliser
- $i$ : Indice de l'epoch actuelle

Si la direction n'est pas prise au hasard alors le robot se basera sur sa **Q table**, en cas d'égalité des valeurs, la direction sera de nouveau prise au hasard.

```python
def pick_direction(state): 
    """
    Random chance to take a random direction to add exploration at the beginning
    Pick a direction based on previous learning, if all the direction are equals in value take a random direction
    :param state: Current position
    :return: New position picked
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


```

### Calcul de Q

Le calcul de Q est basé sur cette formule : 

${\displaystyle Q^{new}(s_{t},a_{t})\leftarrow (1-\alpha )\cdot \underbrace {Q(s_{t},a_{t})} _{\text{old value}}+\underbrace {\alpha } _{\text{learning rate}}\cdot \overbrace {{\bigg (}\underbrace {r_{t}} _{\text{reward}}+\underbrace {\gamma } _{\text{discount factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\text{estimate of optimal future value}}{\bigg )}} ^{\text{learned value}}}$

Avec `eq_first_part` = $ (1-\alpha )\cdot \underbrace {Q(s_{t},a_{t})} _{\text{old value}}$

et `eq_second_part` $= \underbrace {\alpha } _{\text{learning rate}}\cdot \overbrace {{\bigg (}\underbrace {r_{t}} _{\text{reward}}+\underbrace {\gamma } _{\text{discount factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\text{estimate of optimal future value}}{\bigg )}} ^{\text{learned value}}$

```python
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
```



## Questions

Le robot est représenté par un $7$ sur la carte quand elle est affiché en console.

### 1

Après 100 epoch et comportant chacune 100 mouvement du robot, il semble se stabiliser sur le **but 2** valant 50 points.

Ceci est certainement dus au fait qu'il soit plus proche que le **but 1**.

#### Résultats

```pseudocode
current_step =  96
[[   0.    0.    0.    0.    0. 1000.]
 [   7.    0.    0.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]
 [   0.    0.  -10.  -10.  -10.  -10.]
 [   0.    0.  -10.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]]

current_step =  97
[[   0.    0.    0.    0.    0. 1000.]
 [   7.    0.    0.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]
 [   0.    0.  -10.  -10.  -10.  -10.]
 [   0.    0.  -10.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]]

current_step =  98
[[   0.    0.    0.    0.    0. 1000.]
 [   7.    0.    0.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]
 [   0.    0.  -10.  -10.  -10.  -10.]
 [   0.    0.  -10.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]]

current_step =  99
[[   0.    0.    0.    0.    0. 1000.]
 [   7.    0.    0.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]
 [   0.    0.  -10.  -10.  -10.  -10.]
 [   0.    0.  -10.    0.    0.    0.]
 [   0.    0.    0.    0.    0.    0.]]

```

### 2

Pour trouver plus rapidement le but il suffit de se 'ajouter l'inverse (ou l'opposé en fonction de si l'on veut récompenser ou punir) de la distance à la reward de chaque case.

Pour cela on modiefira simplement les récompenses de chaque case.

Carte des rewards en fonctions des distances :

```pseudocode
[[0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667]
 [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667]
 [0.125      0.125      0.125      0.125      0.125      0.125     ]
 [0.1        0.1        0.1        0.1        0.1        0.1       ]
 [0.08333333 0.08333333 0.08333333 0.08333333 0.08333333 0.08333333]
 [0.07142857 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857]]
```

Il ne reste plus qu'à ajouter ces nouveaux poids aux anciens.

```python
def compute_distance_game_map(current_game_map):
    """
    Compute new reward for each tile based on the opposite of the distance from both points
    :param current_game_map:
    :return: new weighted map
    """
    local_game_map = current_game_map.copy()
    for y in range(local_game_map.shape[0]):
        for x in range(local_game_map.shape[1]):
            local_game_map[y, x] = local_game_map[y, x] + 1 / (abs(x - 5) + y + x + abs(y - 1))
    return local_game_map

```

On observe qu'étrangement cette fois le robot converge vers le **but 1**.

Le script est `main_q2.py`.

### 3

Pour avoir un parcours de type **but 1**, **but 2**, retour au **départ**, il faudrait mettre à jour les buts une fois un but atteint, nullifié celui atteint puis récompenser uniquement le suivant et ainsi de suite.