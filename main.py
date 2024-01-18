from game import Game
from Agent import Agent
from AIAgent import DQNAgent
import torch

game = Game()


def train():
    agent1 = Agent("blue", 0)
    agent2 = Agent("darkgreen", 1)
    game.addPlayers(agent1, agent2)
    while True:
        # get state
        state_old = game.getState()

        # get move
        action = [[], []]
        action[0] = agent1.getAction(state_old)
        action[1] = agent2.getAction(state_old)

        # perform move and get new state
        reward, done = game.playStep(action)
        state_new = game.getState()

        agent1.remember(state_old, action[0], reward[0], state_new, done)
        agent2.remember(state_old, action[1], reward[1], state_new, done)

        agent1.replay()
        agent2.replay()

        # train short memory
        # agent1.train_short_memory(state_old, action[0], reward[0], state_new, done)
        # agent2.train_short_memory(state_old, action[1], reward[1], state_new, done)

        # remember
        # agent1.remember(state_old, action[0], reward[0], state_new, done)
        # agent2.remember(state_old, action[1], reward[1], state_new, done)

        if done:
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1
            # agent1.trainLongMemory()
            # agent2.trainLongMemory()

            agent1.epsilon *= agent1.epsilonDecay
            agent2.epsilon *= agent2.epsilonDecay

            agent1.NN.save("weights1.pth")
            agent2.NN.save("weights2.pth")
            print("Continuing with game: ", str(agent1.n_games))

            if agent1.epsilon < agent1.epsilonMin:
                break

            if agent1.n_games % 100 == 0:
                agent1.NN.save("weights1" + str(agent1.n_games) + "games" ".pth")
                agent2.NN.save("weights2" + str(agent2.n_games) + "games" ".pth")


def train2(env, episodes):
    agent1 = Agent("blue", 0)
    agent2 = Agent("darkgreen", 1)
    game.addPlayers(agent1, agent2)
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = [0, 0]

        while not done:
            action = []
            action.append(agent1.getAction(state))
            action.append(agent2.getAction(state))

            next_state, reward, done = env.playStep(action)

            total_reward[0] += reward[0]
            total_reward[1] += reward[1]

            agent1.remember(state, action[0], reward[0], next_state, done)
            agent2.remember(state, action[1], reward[1], next_state, done)
            state = next_state

            agent1.replay()
            agent2.replay()

        agent1.NN.save("weights1.pth")
        agent2.NN.save("weights2.pth")

        print(
            f"Episode: {episode + 1}, Total Reward: {total_reward[0]}, Epsilon: {agent1.epsilon} \n"
            f"Episode: {episode + 1}, Total Reward: {total_reward[1]}, Epsilon: {agent2.epsilon}"
        )


def play2(env, episodes):
    agent1 = Agent("blue", 0)
    agent2 = Agent("darkgreen", 1)
    game.addPlayers(agent1, agent2)
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = [0, 0]

        while not done:
            action = []
            action.append(agent1.getActionNoRandom(state))
            action.append(agent2.getActionNoRandom(state))

            next_state, reward, done = env.playStep(action)

            total_reward[0] += reward[0]
            total_reward[1] += reward[1]

            state = next_state

        print(f"Total Reward: {total_reward[0]}" f"Total Reward: {total_reward[1]}")


def play():
    agent1 = Agent("blue", 0)
    agent2 = Agent("darkgreen", 1)
    game.addPlayers(agent1, agent2)
    while True:
        # get state
        state_old = game.getState()

        # get move
        action = [[], []]
        action[0] = agent1.getActionNoRandom(state_old)
        action[1] = agent2.getActionNoRandom(state_old)

        # perform move and get new state
        reward, done = game.playStep(action)

        if done:
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1

            print("Continuing with game: ", str(agent1.n_games))


def play3(episodes):
    p1 = DQNAgent(17, 8, "blue", 0)
    p2 = DQNAgent(17, 8, "darkgreen", 1)
    game.addPlayers(p1, p2)
    p1.q_network.load_state_dict(torch.load("dqn_model1.pth"))
    p2.q_network.load_state_dict(torch.load("dqn_model2.pth"))
    p1.epsilon = 0
    p2.epsilon = 0
    for _ in range(episodes):
        state = game.reset()
        done = False
        total_reward = [0, 0]

        while not done:
            action = []
            action.append(p1.select_action(state))
            action.append(p2.select_action(state))

            next_state, reward, done = game.playStep(action)

            total_reward[0] += reward[0]
            total_reward[1] += reward[1]

            state = next_state

        print(f"Total Reward: {total_reward[0]}" f"Total Reward: {total_reward[1]}")


if __name__ == "__main__":
    # train()
    # play()

    train2(game, 300)
    # play2(game, 10)

    # play3(10)
