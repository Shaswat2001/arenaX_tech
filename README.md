# ArenaX COOP challenge 

## About the Challenge

In this task, a reinforcement learning algorithm was to be designed with focus on cooperation, exploration and sparse rewards. 

For a successful episode - 
1. Two mice must reach the cheese together for the episode to be considered successful.
2. Some gates block the path to the cheese. To open these gates, at least one mouse must interact with a specific flower.
3. Players must switch between the two mice strategically, helping each other navigate obstacles and progress toward the cheese.

### Environment information 

* Map Size - Grid of size `19x13`
* State Space - A matrix of `19x13x13` where each channel represents different elements of the grid, like mouse, cheese, walls and shrubs encoded.
* Action Space - Integer value between 0 and 9, each mapped to a direction in which a mouse can move and 9 used to switch between the two.
* Reward - 1 if both the mice reaches the cheese and 0 elsewhere. 

### Issues 

* With the sparse reward, a general RL algorithm will have hard time find an optimal solution. 

## Approaches - 

### Behaviour Cloning - 

To address the sparse reward issue, I first applied Behavior Cloning (BC). The goal was to learn a policy by imitating expert demonstrations rather than relying solely on reward signals. Here’s what I did:

1. **Data Collection**: I manually played the game and recorded my actions to create demonstration trajectories. These demonstrations consisted of both desired behaviors and recovery actions when the mouse encountered an undesired scenario.

2. **Training**: A supervised learning model was trained to predict the next action based on the current state, effectively mimicking expert decisions.

3. **Results**: The BC-trained model successfully solved the task with a success rate of 50%.

![Behaviour Cloning Working](resources/bc_working.gif)


However, it did not learn meaningful representations as expected, which was evident in the failure cases observed in this video.

![Behaviour Cloning Failing](resources/bc_not_working.gif)

Why Did It Fail? 🤔

Behavior Cloning has inherent drawbacks that limit its effectiveness in complex RL tasks:

1. 🔍 **Lack of Exploration**: Since BC is a supervised learning approach, the agent only learns from recorded demonstrations and does not explore novel situations. This is problematic in environments where exploration is necessary for success.

2. ⚠️ **Compounding Errors**: If the agent deviates slightly from the expert trajectory, it has no mechanism to recover since it has not seen those states during training.

3. 📊 **Data Dependence**: The model's performance is highly dependent on the quality and diversity of the demonstrations. If the dataset does not cover all possible scenarios, the agent struggles in unseen situations.

4. 🏆 **No Reward Utilization**: Unlike reinforcement learning, BC does not leverage reward signals, meaning it cannot refine its policy based on actual performance outcomes.