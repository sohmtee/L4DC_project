############ modified full algorithm with KUCBVI 10/15 (done with Maheed)    #################### MOST RECENT

import numpy as np
import matplotlib.pyplot as plt
# import torch
from scipy.linalg import eigh

# Define the grid world parameters
grid_size = (4, 4)
treasure_pos = (3, 3)
lightning_pos = (2, 1)
mountain_pos = (1, 2)
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)
num_states = grid_size[0] * grid_size[1]
horizon = 30

num_epochs = 5000
num_classes = 4

learning_rate = 0.01
T = 10000
epsilon = 0.2
alpha = 0.001

# Define the transition probabilities
prob_intended = 0.91
prob_other = 0.03

# Define the movement deltas for each action
action_deltas = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

delta = 0.99                                # parameter controlling the confidence level.
B = 0.01                                       # Upper bound on the norm of w_star.
eta = np.exp(-4*B)/2                        # parameter controlling the exploration-exploitation balance.
C = np.log(num_classes * np.exp(2*B))       # constant related to the log of the number of classes and B.

omega_star = np.array([[ 0.6589],
        [ 0.2591],
        [-0.8223],
        [-4.6706]])

omega_star_norm = np.linalg.norm(omega_star)/16

# Function to check if a position is valid
def is_valid_pos(pos):
    if pos == mountain_pos:
        return False
    if 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]:
        return True
    return False

# Function to get the next state given the current state and action
def get_next_state(state, action):
    if state in [treasure_pos, lightning_pos]:
        return state
    intended_pos = (state[0] + action_deltas[action][0], state[1] + action_deltas[action][1])
    if not is_valid_pos(intended_pos):
        intended_pos = state
    next_state = intended_pos
    rand = np.random.rand()
    if rand <= prob_intended:
        next_state = intended_pos
    else:
        # Calculate remaining probability to distribute among other actions
        remaining_prob = 1 - prob_intended
        other_prob = remaining_prob / 3
        for other_action in actions:
            if other_action != action:
                other_pos = (state[0] + action_deltas[other_action][0], state[1] + action_deltas[other_action][1])
                if not is_valid_pos(other_pos):
                    other_pos = state
                rand -= other_prob
                if rand <= 0:
                    next_state = other_pos
                    break
    return next_state

# Function to calculate the Manhattan distance
def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

# Function to generate a sample trajectory
def generate_trajectory(theta, horizon):
    trajectory = []
    start_state = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
    state = start_state
    for _ in range(horizon):
        state_index = state[0] * grid_size[1] + state[1]
        action_probs = softmax_policy(theta, state_index)
        # print('action probs', action_probs)
        action_index = select_action(action_probs)
        action = actions[action_index]
        trajectory.append((state, action))
        next_state = get_next_state(state, action)
        state = next_state
        if state in treasure_pos:
            break
    trajectory.append((state, action))
    return trajectory

def compute_feature_vector(trajectory, class_index, feature_dim=1):
    original_feature = manhattan_distance(trajectory[-1][0], treasure_pos)
    feature_vector = np.zeros(num_classes * feature_dim)
    start_idx = class_index * feature_dim
    end_idx = start_idx + feature_dim
    feature_vector[start_idx:end_idx] = original_feature
    return feature_vector

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def softmax_policy(theta, state_index):
    return softmax(theta[state_index])

# select an action based on the current policy
def select_action(policy):
    return np.random.choice(len(actions), p=policy)

def compute_sigma_D_t(trajectories, t, d=1, regularization=0.01):
    feature_vec_dim = num_classes * d                              
    sigma_D_t = np.zeros((feature_vec_dim, feature_vec_dim))
    feature_vectors = np.zeros((num_classes, len(trajectories), feature_vec_dim))  
    for i in range(num_classes):
        feature_vectors[i,:,:] = np.array([compute_feature_vector(traj, i, feature_dim=d) for traj in trajectories])
    for i in range(len(trajectories)):
        for j in range(num_classes):
            for l in range(num_classes):
                diff = feature_vectors[j,i] - feature_vectors[l,i]
                sigma_D_t += np.outer(diff, diff)
    sigma_D_t /= (t * num_classes**2)
    sigma_D_t += regularization * np.eye(feature_vec_dim)
    # print('sigma D', sigma_D_t)
    return sigma_D_t

def compute_estimated_reward(w, trajectory):
    logits = np.zeros(num_classes)
    for i in range(num_classes):
        phi_i = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w.flatten(), phi_i)
    probs = softmax(logits.flatten())
    return np.sum(np.arange(num_classes) * probs)

def compute_adjusted_reward(trajectory, w, t, sigma_D_t):
    rew_est = compute_estimated_reward(w, trajectory)
    lambda_min = np.min(eigh(sigma_D_t, eigvals_only=True))
    # print('lambda min', lambda_min)
    bonus = ((4 * num_classes * np.exp(4 * B)) / 
            (eta * lambda_min) * 
            np.sqrt((C**2 / (2 * t)) * np.log(4 / delta))) * (1/20)
    # print(f'uncertainty term at iteration {t}', bonus)
    optimistic_reward = min(rew_est + bonus, num_classes - 1)
    return optimistic_reward

def compute_gradient(theta, trajectories, rewards):
    grad = np.zeros_like(theta)
    for traj, reward in zip(trajectories, rewards):
        for state, action in traj:
            state_index = state[0] * grid_size[1] + state[1]
            action_probs = softmax_policy(theta, state_index)
            action_index = actions.index(action)              # get the index of the action
            grad[state_index, action_index] += reward * (1 - action_probs[action_index])
            for a in range(num_actions):
                if a != action_index:
                    grad[state_index, a] -= reward * action_probs[a]
    return grad


# def compute_gradient(theta, trajectories, rewards):
#     grad = np.zeros_like(theta)
#     num_trajectories = len(trajectories)
    
#     # Expectation over trajectories
#     for trajectory, reward in zip(trajectories, rewards):
#         # Sum over time steps
#         for state, action in trajectory:
#             state_index = state[0] * grid_size[1] + state[1]
            
#             # Compute policy probabilities π_θ(a|s)
#             action_probs = softmax_policy(theta, state_index)
#             action_index = actions.index(action)
            
#             # Compute gradient of log-policy for all actions
#             for a in range(num_actions):
#                 if a == action_index:
#                     # For taken action: (1 - π_θ(a_t|s_t))
#                     grad[state_index, a] += reward * (1 - action_probs[a])
#                 else:
#                     # For other actions: -π_θ(a|s_t)
#                     grad[state_index, a] -= reward * action_probs[a]
    
#     return grad

# Function to calculate P(y_τ = i)
def feedback_prob(w_star, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w_star.flatten(), phi)
    return softmax(logits.flatten())

# Function to calculate the estimation error between omega* and omega_hat
def estimation_error(w_star, w_hat):
    return np.linalg.norm(w_star - w_hat)

def compute_loss_gradient(w_estimate, trajectories, y_true, n):
    grad = np.zeros_like(w_estimate)
    
    for i in range(n):
        logits = np.zeros(num_classes)
        # Compute logits for all classes
        for j in range(num_classes):
            feature_j = compute_feature_vector(trajectories[i], j)
            logits[j] = np.dot(w_estimate.flatten(), feature_j)

        y_pred = softmax(logits.flatten())
        feature_y = compute_feature_vector(trajectories[i], np.argmax(y_true[i]))           ## extracting the true class
        for j in range(num_classes):
            feature_j = compute_feature_vector(trajectories[i], j)
            grad += y_pred[j] * (feature_y - feature_j).reshape(-1, 1)
    return -grad / n

def project_grad(w_est, B):
    norm_w = np.linalg.norm(w_est)
    if norm_w > B:
        w_est = (B * w_est) / norm_w
    return w_est

## computing true reward
def compute_true_reward(w_star, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w_star.flatten(), phi)
    probs = softmax(logits.flatten())
    return np.sum(np.arange(num_classes) * probs)


## for visualization

def plot_grid_world_policy(theta):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create 4x4 grid
    for i in range(5):
        ax.axhline(i, color='black', lw=1)
        ax.axvline(i, color='black', lw=1)
    
    # Dictionary to map actions to arrows
    action_arrows = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→'
    }
    
    # For each cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state_index = i * grid_size[1] + j
            
            # Skip mountain position
            if (i, j) == mountain_pos:
                ax.add_patch(plt.Rectangle((j, 3-i), 1, 1, fill=True, color='gray'))
                ax.text(j+0.5, 3-i+0.5, 'M', ha='center', va='center', fontsize=12)
                continue
                
            # Special states
            if (i, j) == treasure_pos:
                ax.add_patch(plt.Rectangle((j, 3-i), 1, 1, fill=True, color='gold', alpha=0.3))
                ax.text(j+0.5, 3-i+0.5, 'T', ha='center', va='center', fontsize=12)
            elif (i, j) == lightning_pos:
                ax.add_patch(plt.Rectangle((j, 3-i), 1, 1, fill=True, color='red', alpha=0.3))
                ax.text(j+0.5, 3-i+0.5, 'L', ha='center', va='center', fontsize=12)
            
            # Get action probabilities for this state
            probs = softmax_policy(theta, state_index)
            max_prob_index = np.argmax(probs)
            max_prob = probs[max_prob_index]
            max_action = actions[max_prob_index]
            
            # Add arrow and probability
            ax.text(j+0.5, 3-i+0.7, action_arrows[max_action], 
                   ha='center', va='center', fontsize=20)
            ax.text(j+0.5, 3-i+0.3, f'{max_prob:.2f}', 
                   ha='center', va='center', fontsize=10)
    
    # Add legend
    ax.text(-0.5, 4.2, 'T: Treasure, L: Lightning, M: Mountain', fontsize=10)
    ax.text(-0.5, 3.8, '↑↓←→: Most probable action', fontsize=10)
    ax.text(-0.5, 3.5, 'Number: Action probability', fontsize=10)
    
    plt.title('Grid World Policy (Most Probable Actions)')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



num_runs = 2

reward_all_runs = np.zeros((num_runs, T))

for run in range(num_runs):
    print(f'Run {run+1}/{num_runs}')
    theta = np.random.rand(num_states, num_actions)
    w_estimate = np.random.rand(num_classes,1)
    theta_norm_diffs = []
    last_5_theta_diffs = []

    # Plot initial policy
    plt.figure(figsize=(8, 8))
    plot_grid_world_policy(theta)
    plt.title(f'Initial Policy (Run {run+1})')
    plt.show()

    for t in range(T):
        if t % 100 == 0:
            print(f'Outer Loop {t}/{T}')

        while True:
            prev_theta = np.copy(theta)

            trajectories = [generate_trajectory(theta, horizon) for _ in range(50)]
            sigma_D_t = compute_sigma_D_t(trajectories, t+1, d=1)
            rewards = [compute_adjusted_reward(traj, w_estimate, t+1, sigma_D_t) for traj in trajectories]
            # print(f'adjusted reward at iteration {t+1} is', rewards)
            
            grad = compute_gradient(theta, trajectories, rewards)
            theta += alpha * grad
            theta_norm_diff = np.linalg.norm(theta - prev_theta)
            # print(f'theta norm diff at iteration {t+1}', theta_norm_diff)
            theta_norm_diffs.append(theta_norm_diff)
            # print('theta norm differences', theta_norm_diffs)
            
            last_5_theta_diffs.append(theta_norm_diff)
            if len(last_5_theta_diffs) > 5:
                last_5_theta_diffs.pop(0)

            if len(last_5_theta_diffs) == 5:
                avg_theta_diff = np.mean(last_5_theta_diffs)
                # print(f'avg theta diff at iteration {t+1}', avg_theta_diff)

                if avg_theta_diff <= epsilon:
                    # print('converged theta over the last 5 steps')
                    break


            # if theta_norm_diff <= epsilon:
            #     # print('Converged theta')
            #     break

        # Generate new trajectories using updated theta
        eval_trajectories = [generate_trajectory(theta, horizon) for _ in range(50)]
        true_rewards = [compute_true_reward(omega_star, traj) for traj in eval_trajectories]
        # print('true rews', np.linalg.norm(true_rewards))
        reward_all_runs[run, t-1] = np.mean(true_rewards)                                                       ## store mean reward for this episode
        # print(f'rewards all at iteration {t+1}', reward_all_runs)

    # Plot final policy for this run
    plt.figure(figsize=(8, 8))
    plot_grid_world_policy(theta)
    plt.title(f'Final Policy (Run {run+1})')
    plt.show()

average_rewards = np.mean(reward_all_runs, axis=0)
std_rewards = np.std(reward_all_runs, axis=0)

plt.plot(range(T), average_rewards)
plt.fill_between(range(T),
                 average_rewards - 2 * std_rewards,
                 average_rewards + 2 * std_rewards,
                 color='b', alpha=0.2, label="95% Confidence Interval")
plt.xlabel('Number of Episodes')
plt.ylabel('Average True Reward')
plt.title('Average True Reward over Multiple Runs')
plt.show()

# # Plot the norm differences over time
plt.plot(theta_norm_diffs)
plt.xlabel('Iterations')
plt.ylabel(r'$||\theta_{t+1} - \theta_t||$')
plt.title('Theta Update Norm Differences Over Iterations')
plt.show()