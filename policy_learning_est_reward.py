import numpy as np
import matplotlib.pyplot as plt
# import torch

# Define the grid world parameters
grid_size = (4, 4)
treasure_pos = (3, 3)
lightning_pos = (2, 1)
mountain_pos = (1, 2)
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)
num_states = grid_size[0] * grid_size[1]
horizon = 20

num_epochs = 5000
num_classes = 4

learning_rate = 0.0001
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

delta = 0.99                         
B = 0.01                             
eta = np.exp(-4*B)/2                 
C = np.log(num_classes * np.exp(2*B))

omega_star = np.array([[ 0.6589],
        [ 0.2591],
        [-0.8223],
        [-4.6706]])

omega_star_norm = np.linalg.norm(omega_star)/16

def is_valid_pos(pos):
    if pos == mountain_pos:
        return False
    if 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]:
        return True
    return False

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

def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

def generate_trajectory(theta, horizon):
    trajectory = []
    start_state = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
    state = start_state
    for _ in range(horizon):
        state_index = state[0] * grid_size[1] + state[1]
        action_probs = softmax_policy(theta, state_index)
        # print(f'action probs at iteration {t+1}', action_probs)
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

def select_action(policy):
    return np.random.choice(len(actions), p=policy)

def compute_reward(w, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w.flatten(), phi)
    probs = softmax(logits.flatten())
    # print('probs of estimated reward is', probs)
    return np.sum(np.arange(num_classes) * probs)

def compute_gradient(theta, trajectories, rewards):
    grad = np.zeros_like(theta)
    for traj, reward in zip(trajectories, rewards):
        for state, action in traj:
            state_index = state[0] * grid_size[1] + state[1]
            action_probs = softmax_policy(theta, state_index)
            action_index = actions.index(action)              # get the index of the action
            # print('action index is', action_index)
            # print(f"Action probabilities for state {state_index}: {softmax_policy(theta, state_index)}")
            grad[state_index, action_index] += reward * (1 - action_probs[action_index])
            for a in range(num_actions):
                if a != action_index:
                    grad[state_index, a] -= reward * action_probs[a]
    return grad

def feedback_prob(w_star, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w_star.flatten(), phi)
    return softmax(logits.flatten())

def estimation_error(w_star, w_hat):
    return np.linalg.norm(w_star - w_hat)

def compute_true_reward(w_star, trajectory):
    logits = np.zeros((num_classes))
    for i in range(num_classes):
        phi = compute_feature_vector(trajectory, i)
        logits[i] = np.dot(w_star.flatten(), phi)
    probs = softmax(logits.flatten())
    # print(f'probs of true reward at iteration {t+1}', probs)
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



num_runs = 10

reward_all_runs = np.zeros((num_runs, T))

for run in range(num_runs):
    print(f'Run {run+1}/{num_runs}')
    theta = np.random.rand(num_states, num_actions)
    theta_norm_diffs = []
    w_estimate = np.random.rand(num_classes, 1)
    last_5_theta_diffs = []

    # Plot initial policy
    plt.figure(figsize=(8, 8))
    plot_grid_world_policy(theta)
    plt.title(f'Initial Policy (Run {run+1})')
    plt.show()
    
    for t in range(T):
        if t % 100 == 0:
            print(f'Episode {t}/{T}')

        #     # Plot policy every 100 episodes
        #     plt.figure(figsize=(8, 8))
        #     plot_grid_world_policy(theta)
        #     plt.title(f'Policy at Episode {t} (Run {run+1})')
        #     plt.show()
        while True:
            prev_theta = np.copy(theta)
            
            trajectories = [generate_trajectory(theta, horizon) for _ in range(200)]
            rewards = [compute_reward(w_estimate, traj) for traj in trajectories]
            # print(f'estimated rewards at iteration {t+1} is', rewards)

            # Compute gradient and update theta
            grad = compute_gradient(theta, trajectories, rewards)
            # print(f'grad at iteration {t+1} is', np.linalg.norm(grad))
            theta += alpha * grad

            ## for a single state
            # print(f'softmax policy of theta at iteration {t+1} for state 1', softmax_policy(theta, 1))
            ## for all states
            # for state_index in range(num_states):
            #     print(f'softmax policy of theta at iteration {t+1} for state {state_index}: ', softmax_policy(theta, state_index))
            theta_norm_diff = np.linalg.norm(theta - prev_theta)
            theta_norm_diffs.append(theta_norm_diff)

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
            #     break

        eval_trajectories = [generate_trajectory(theta, horizon) for _ in range(50)]
        true_rewards = [compute_true_reward(omega_star, traj) for traj in eval_trajectories]
        # print(f'sum of true rewards at iteration {t+1} is', np.sum(true_rewards))
        reward_all_runs[run, t-1] = np.mean(true_rewards)
        # print(f'reward all runs at iteration {t+1} is', len(reward_all_runs))

    # Plot final policy for this run
    plt.figure(figsize=(8, 8))
    plot_grid_world_policy(theta)
    plt.title(f'Final Policy (Run {run+1})')
    plt.show()


avg_rewards = np.mean(reward_all_runs, axis=0)
print('avg reward is', len(avg_rewards))
std_rewards = np.std(reward_all_runs, axis=0)

plt.plot(range(T), avg_rewards, label='Average True Reward')
plt.fill_between(range(T),
                 avg_rewards - 0.5 * std_rewards,
                 avg_rewards + 0.5 * std_rewards,
                 color='b', alpha=0.2, label="Confidence Interval")
plt.xlabel('Episodes')
plt.ylabel('Average True Reward')
plt.title('Average Reward vs. Episodes')
plt.legend()
plt.show()

# plt.plot(theta_norm_diffs)
# plt.xlabel('Iterations')
# plt.ylabel(r'$||\theta_{t+1} - \theta_t||$')
# plt.title('Theta Update Norm Differences Over Iterations')
# plt.grid(True)
# plt.show()
