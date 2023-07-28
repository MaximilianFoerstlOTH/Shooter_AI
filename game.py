import pygame 
import time
from Agent import Agent, Border
import numpy

class Game:
    characters = []
    def __init__(self, agent1 : Agent, agent2 : Agent):

        self.height = 1000
        self.width = 1000
        self.timeUnit = 0
        #self.screen = None
        self.player1 = agent1
        self.player2 = agent2
        self.reward = [0,0]
        self.border = Border(0, 0, self.width, self.height)
        self.player1.setEnemy(self.player2)
        self.player2.setEnemy(self.player1)
        self.characters.append(self.player1)
        self.characters.append(self.player2)
        self.startGame()

    def playStep(self, screen) ->  bool:
        for i in range(len(self.characters)):
            done = False
            #screen_data = pygame.surfarray.array3d(screen)
            #self.characters[i].step(screen_data, self.reward[i], False)
            self.characters[i].getAction(screen, self.reward[i], False)
            playercollided = self.characters[i].checkCollisionWithEnemyBullet() or self.border.checkCollision(self.characters[i])
            #self.border.checkCollision(self.characters[i])
            if playercollided:
                self.reward[i] = -100
                self.reward[i-1] = 100
                done = True
            else: 
                self.reward[i] = 1
            #Every 10 frames:
            if self.timeUnit == 9:
                self.border.decreaseRadius(15)
            self.timeUnit += 1
            self.timeUnit %= 10
            
            return done


    def startGame(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            #draw to Screen
            screen.fill((134, 126, 255))
            for character in self.characters:
                character.draw(screen)
                for bullet in character.bullets:
                    bullet.draw(screen)
            self.border.draw(screen)

            self.surface = pygame.surfarray.array3d(pygame.display.get_surface())
            self.height, self.width, self.channels = self.surface.shape

            #Play step
            if self.playStep(self.surface) == True:
                running = False


            # flip() the display to put your work on screen
            pygame.display.flip()
            #Set FPS
            clock.tick(60)

        pygame.quit()



#game = Game()