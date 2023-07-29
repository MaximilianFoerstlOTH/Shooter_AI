from game import Game
from Agent import Agent




def train():
    game = Game()
    agent1 = Agent("blue" ,0)
    agent2 = Agent("darkgreen" , 1)
    game.addPlayers(agent1, agent2)
    while True:
        #get state
        state_old = game.getState()

        #get move
        action = [[],[]]
        action[0] = agent1.getAction(state_old)
        action[1] = agent2.getAction(state_old)

        #perform move and get new state
        reward , done = game.playStep(action)
        state_new = game.getState()

        #train short memory
        agent1.train_short_memory(state_old, action[0], reward[0], state_new, done)
        agent2.train_short_memory(state_old, action[1], reward[1], state_new, done)

        #remember
        agent1.remember(state_old, action[0], reward[0], state_new, done)
        agent2.remember(state_old, action[1], reward[1], state_new, done)

        if done:
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1
            agent1.trainLongMemory()
            agent2.trainLongMemory()
            agent1.NN.save("weights1.pth")
            agent2.NN.save("weights2.pth")

            

if __name__ == "__main__":
    train()