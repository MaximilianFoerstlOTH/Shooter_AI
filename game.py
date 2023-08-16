import pygame 
import time
from Agent import Agent, Border
import numpy
from const import width
pygame.init()

class Game:
    characters = []
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.height = width
        self.width = width
        self.timeUnit = 0
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.shootCooldown = [0,0]
        #self.screen = None
        self.player1 = None
        self.player2 = None
        self.border = Border(0, 0, self.width, self.height)
        #self.player1.setEnemy(self.player2)
        #self.player2.setEnemy(self.player1)
        #self.characters.append(self.player1)
        #self.characters.append(self.player2)
        #self.startGame()

    def addPlayers(self, player1 : Agent, player2 : Agent):
        self.player1 = player1
        self.player2 = player2
        self.player1.setEnemy(self.player2)
        self.player2.setEnemy(self.player1)
        self.characters.append(self.player1)
        self.characters.append(self.player2)


    def playStep(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        done = False
        reward = [0,0]
        for i in range(len(self.characters)):

            for bullet in self.characters[i].bullets:
                if bullet.collisionWithPlayer():
                    print("Player ", i+1 , self.characters[i].colour , " got hit by enemy bullet")
                    reward[i] = -100
                    reward[i-1] = 20
                    done = True
                    return reward,done
            #if self.characters[i].checkCollisionWithEnemyBullet():
                            #if self.player1.checkCollisionWithEnemyBullet() or self.player2.checkCollisionWithEnemyBullet():
                #print("Player ", i+1 , " got hit by enemy bullet")
                #reward[i] = -100
                #reward[i-1] = 100
                #done = True
                #return reward,done
            
            if self.border.checkCollision(self.characters[i]):
                reward[i] = -100
                done = True
                return reward,done
            
            else: 
                reward[i] = 1
            #Move player
            movex, movey, s, sx,sy = action[i]
            self.characters[i].Move(movex, movey)
            for bullet in self.characters[i].bullets:
                bullet.moveBullet()
            if self.shootCooldown[i] > 60:
                self.characters[i].Shoot(s, sx, sy)
                if s > 0:
                    self.shootCooldown[i] = 0
            else:
                self.shootCooldown[i] += 1
            
            #Every 10 frames:
            if self.timeUnit == 9:
                self.border.decreaseRadius(2)
            self.timeUnit += 1
            self.timeUnit %= 10


        self.render()
        self.clock.tick(60)
        #print("FPS: ", self.clock.get_fps())
            
        return reward,done


    def getState(self):
        p1Nx, p1Ny, p1Ndx, p1Ndy = self.player1.findNearestBullet()
        p2Nx, p2Ny, p2Ndx, p2Ndy = self.player2.findNearestBullet()
        return numpy.array([self.player1.x, self.player1.y, self.player1.moveDir.x, self.player1.moveDir.y , p1Nx, p1Ny, p1Ndx, p1Ndy , self.player2.x, self.player2.y,self.player2.moveDir.x, self.player2.moveDir.y , p2Nx, p2Ny, p2Ndx, p2Ndy, self.border.width])

    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.border = Border(0, 0, self.width, self.height)
        self.render()
        self.timeUnit = 0
        #return self.getState()

    def render(self):
        #   draw to Screen
        self.screen.fill((134, 126, 255))
        for character in self.characters:
            character.draw(self.screen)
            for bullet in character.bullets:
                bullet.draw(self.screen)
        self.border.draw(self.screen)

        self.surface = pygame.surfarray.array3d(pygame.display.get_surface())
        self.height, self.width, self.channels = self.surface.shape
        pygame.display.flip() 

    # def startGame(self):
    #     screen = pygame.display.set_mode((self.width, self.height))
    #     clock = pygame.time.Clock()
    #     running = True

    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False

    #         #draw to Screen
    #         screen.fill((134, 126, 255))
    #         for character in self.characters:
    #             character.draw(screen)
    #             for bullet in character.bullets:
    #                 bullet.draw(screen)
    #         self.border.draw(screen)

    #         self.surface = pygame.surfarray.array3d(pygame.display.get_surface())
    #         self.height, self.width, self.channels = self.surface.shape

    #         #Play step
    #         if self.playStep() == True:
    #             running = False

    #         # flip() the display to put your work on screen
    #         pygame.display.flip()
    #         #Set FPS
    #         clock.tick(60)

    #     pygame.quit()



#game = Game()