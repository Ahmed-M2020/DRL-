# DRL-
Assignments of Deep Reinforcement Learning 
Introduction: 
This report summarizes my completion of the missing parts in the training loop for the Pong game using the DQN algorithm. It also discusses the architecture of the DQNNetwork implemented for the assignment. Furthermore, it presents the training results and performance evaluation of the Pong game agent.
Training Loop(MlpMinigridPolicy): 
In the training loop for the MlpMinigridPolicy, I completed the missing parts by initializing a replay buffer and storing the state, action, reward, next state, and dones in the buffer. I then sampled a batch of states, actions, rewards, next states, and dones from the buffer. Using the MlpMinigrid Policy, I obtained the Q-values (Q1) for the state batch. By using the target network, I obtained the Q-values (Q2) for the next states. I calculated the targets for training by combining the rewards with the discounted maximum Q-values of the next states, considering the dones. Finally, I computed the predictions of the best actions by multiplying the Q-values (Q1) by the corresponding actions and summing them along the action dimension.
buffer.add(state, a, r, next_state, done)


      states, actions, rewards, next_states, dones = buffer.sample_batch()      
      Q1 = dqn(states)
      with torch.no_grad():
          Q2 = dqn_target(next_states).detach()
      targets = rewards + gamma * torch.max(Q2, dim=1)[0] * (1 - dones)
      predictions = torch.sum(Q1 * actions,dim=1)


DQNNetwork Architecture: 
The DQNNetwork_atari architecture includes two convolutional layers followed by two linear layers. The first convolutional layer has 4 input channels, 16 output channels, an 8x8 kernel, and a stride of 4. The second convolutional layer has 16 input channels, 32 output channels, a 4x4 kernel, and a stride of 2. The first linear layer is (32*9*9,256) and the last linear layer has 256 units.


Training Loop(MlpMinigridPolicy): 

In the training loop for the DQNNetwork_atari, I completed the missing parts by performing the following steps:
Sampled a minibatch randomly from the replay buffer using buffer.sample_batch(minibatch_size). This provided the state batch, action batch, reward batch, next state batch, and done batch.
Computed the Q-values (Q1) for the state batch by passing it through the DQN network using Q1 = dqn(state_batch).
Used the target network to obtain the Q-values (Q2) for the next state batch by detaching the gradients using Q2 = dqn_target(next_state_batch).detach().
Computed the targets for training by combining the rewards with the discounted maximum Q-values of the next states, considering the dones. This was done using the formula targets = rewards + gamma * torch.max(Q2, dim=1)[0] * (1 - dones).
Computed the predictions for training by multiplying the Q-values (Q1) by the corresponding actions and summing them along the action dimension using predictions = torch.sum(Q1 * actions, dim=1).
Training Results and Performance Evaluation: 
In the first training loop, the cumulative reward achieved was 0.8407 over 37,964 frames. The cumulative loss started at 0.27 and decreased to 0.17.
In the subsequent training loops, I made adjustments to the network and hyperparameters in an attempt to improve the agent's performance. However, despite these modifications, I was unable to surpass the highest reward of 20.64 that I previously achieved and submitted on the server.
Conclusion: 
The training loop for the MlpMinigridPolicy was successfully completed by initializing a replay buffer, sampling batches of states, actions, rewards, next states, and dones, and computing targets and predictions. The DQNNetwork_atari architecture consisted of two convolutional layers followed by a linear layer. Although attempts were made to improve the agent's performance in subsequent training loops, the achieved rewards did not surpass the highest recorded reward of 20.64.


