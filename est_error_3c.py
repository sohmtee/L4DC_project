######## est_error_3c.py  ################

import numpy as np
import matplotlib.pyplot as plt

# Define the grid world parameters
grid_size = (4, 4)
treasure_pos = (3, 3)
lightning_pos = (2, 1)
mountain_pos = (1, 2)
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)
num_states = grid_size[0] * grid_size[1]
horizon = 6

num_classes = 4

learning_rate = 0.001
# T = 5000
B = 1

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

omega_star = np.array([[ 0.6589],
        [ 0.2591],
        [-0.8223],
        [-4.6706]])

omega_star = omega_star/16

print('norm of w star', np.linalg.norm(omega_star))

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
        rand -= prob_intended
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

# trajectory from  a uniform policy
def generate_trajectory(horizon):
    trajectory = []
    start_state = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
    state = start_state
    for _ in range(horizon):
        action_index = np.random.choice(len(actions))
        action = actions[action_index]
        trajectory.append((state, action))
        next_state = get_next_state(state, action)
        state = next_state
        if state in treasure_pos:
            break
    trajectory.append((state, -1))                          ## -1 default for actions since it is not being used 
    return trajectory

def compute_feature_vector(trajectory, class_index, feature_dim=1):
    original_feature = manhattan_distance(trajectory[-1][0], treasure_pos)
    feature_vector = np.zeros(num_classes * feature_dim)
    start_idx = class_index * feature_dim
    end_idx = start_idx + feature_dim
    feature_vector[start_idx:end_idx] = original_feature
    norm = np.linalg.norm(feature_vector)
    if norm > 1:
        feature_vector = feature_vector / norm
    return feature_vector

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def softmax_policy(theta, state_index):
    return softmax(theta[state_index])

def select_action(policy):
    return np.random.choice(len(actions), p=policy)

# Function to calculate P(y_τ = i)
def feedback_prob(w_star, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w_star.flatten(), phi)
    return softmax(logits.flatten())

def estimation_error(w_star, w_hat):
    return np.linalg.norm(w_star - w_hat)

def negative_log_likelihood(w, trajectories, feedbacks, n):
    w = w.reshape(num_classes, 1) 
    total_loss = 0

    for i in range(n):
        logits = np.zeros(num_classes)
        for j in range(num_classes):
            phi_j = compute_feature_vector(trajectories[i], j)
            logits[j] = np.dot(w.T, phi_j)

        max_logit = np.max(logits)  # For numerical stability
        log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))

        true_feedback_index = np.argmax(feedbacks[i])
        phi_y = compute_feature_vector(trajectories[i], true_feedback_index)
        
        log_numerator = np.dot(w.T, phi_y) - max_logit
        total_loss -= (log_numerator - log_sum_exp)

    return total_loss / n

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

def project_grad(w_est, B, projection_counter):
    norm_w = np.linalg.norm(w_est)
    if norm_w > B:
        w_est = (B * w_est) / norm_w
        projection_counter += 1
    return w_est, projection_counter



def inner_loop_gd(trajectories, feedbacks, n, epsilon=1e-8, max_iter=1000):
    w = np.random.rand(num_classes, 1)
    # w = np.zeros((num_classes, 1))
    if np.linalg.norm(w, ord=2) > B:
        w = (B / np.linalg.norm(w, ord=2)) * w
    
    iter_count = 0
    inner_projection_counter = 0
    
    while iter_count < max_iter:
        prev_w = np.copy(w)
        
        grad = compute_loss_gradient(w, trajectories, feedbacks, n)
        
        w -= learning_rate * grad

        w_norm = np.linalg.norm(w)
        if w_norm > B:
            w = (B * w) / w_norm
            inner_projection_counter += 1
        
        # if np.linalg.norm(w - prev_w) < epsilon:
        #     print('norm difference', np.linalg.norm(w - prev_w))
        #     print(f"Converged after {iter_count} iterations")
        #     break
            
        iter_count += 1
    
    if iter_count == max_iter:
        print(f"Inner loop reached maximum iterations ({max_iter}) without converging")
    
    return w, inner_projection_counter


def collect_samples(num_samples):
    trajectories = []
    feedbacks = []

    for _ in range(num_samples):
        trajectory = generate_trajectory(horizon)
        trajectories.append(trajectory)
        feedback_probs = feedback_prob(omega_star, trajectory)
        feedback_score = np.random.choice(num_classes, p=feedback_probs.flatten())
        y_true = np.eye(num_classes)[feedback_score].flatten()
        feedbacks.append(y_true)

    return trajectories, feedbacks


T = 1000

errors = []

current_size = T

trajectories, feedbacks = collect_samples(T)
w_estimate, projection_count = inner_loop_gd(trajectories, feedbacks, current_size)
error = estimation_error(omega_star, w_estimate)
print(f"Estimation Error at {current_size} samples: {error}")
errors.append(error)

# # plt.plot(batch_sizes, errors)
# plt.plot(current_size, errors)
# plt.xlabel('Number of samples')
# plt.ylabel('Estimation Error |ω* - ω_hat_t|')
# plt.title('Estimation Error vs. Number of Samples')
# plt.show()


