from game import Game
from Agent import Agent


game = Game()
agent1 = Agent("blue", 0)
agent2 = Agent("darkgreen", 1)
game.addPlayers(agent1, agent2)


def train():
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


if __name__ == "__main__":
    # train()
    # play()

    train2(game, 300)
    # play2(game, 10)
