import math
import pygame
import time
from DQLearning import DQN, Trainer
from torch.autograd import Variable
from collections import deque
import torch
import random
import os 

MAX_MEMORY = 10000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    x = 0
    y = 0
    width = 50
    height = 50
    playerspeed = 5

    def __init__(self, colour , playernumber):

        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.NN = DQN()
        if torch.cuda.is_available():
            #Macht das Sinn?
            self.NN = self.NN.cuda()
        if os.path.isfile('./model/weights' + str(playernumber+1) + '.pth'):
            print("Loading weights")
            self.NN.load_state_dict(torch.load('./model/weights' + str(playernumber+1) + '.pth'))
        self.trainer = Trainer(self.NN, lr=LR, gamma=self.gamma)

        self.colour = colour

        self.x = 0
        self.y = 0
        self.moveDir = pygame.math.Vector2(0, 0)

        self.playerNumber = playernumber
        self.bullets = []
        
        self.reset()
        
        self.collider = None
        self.enemy = None
        self.moveDir = pygame.math.Vector2(0, 0)
        #self.shotTime = time.time()


    def getAction(self, input):
        self.epsilon = 80 - self.n_games
        final_inputs = [0,0,0,0,0]


        if random.randint(0, 120) < self.epsilon:
            movex = (random.random() - 0.5) *2
            movey = (random.random() - 0.5) *2



            shoot = random.randint(0, 1)
            shootx =(random.random() - 0.5) *20
            shooty =(random.random() - 0.5) *20
            final_inputs[0] = movex
            final_inputs[1] = movey
            if shoot == 0:
                final_inputs[2] = 0
            else:
                final_inputs[2] = 1
            final_inputs[3] = shootx
            final_inputs[4] = shooty

            return final_inputs

        else:
            return self.getActionNoRandom(input)
        #    frameTensor = torch.Tensor(frame)

        #    if torch.cuda.is_available():
        #        frameTensor = frameTensor.cuda()
            
                
        #    inputs = self.NN(frameTensor)

        #    final_inputs[0] = inputs[0][0]
        #    final_inputs[1] = inputs[0][1]
        #    final_inputs[2] = inputs[0][2]
        #    final_inputs[3] = inputs[0][3]
        #    final_inputs[4] = inputs[0][4]

        #return final_inputs
        #self.Move(self.inputs[0][0], self.inputs[0][1], self.inputs[0][2], self.inputs[0][3])
        #self.Shoot(self.inputs[0][4], self.inputs[0][5], self.inputs[0][6])

    def getActionNoRandom(self, input):
        frameTensor = torch.Tensor(input)
        final_inputs = [0,0,0,0,0]
        if torch.cuda.is_available():
            frameTensor = frameTensor.cuda()
            
                
        inputs = self.NN(frameTensor)

        final_inputs[0] = inputs[0]
        final_inputs[1] = inputs[1]
        final_inputs[2] = inputs[2]
        final_inputs[3] = inputs[3]
        final_inputs[4] = inputs[4]
        return final_inputs
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)        
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def MoveWitKey(self):
        #Player Movement
        self.moveDir = pygame.math.Vector2(0, 0)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.moveDir.x -= 1
        if keys[pygame.K_d]:
            self.moveDir.x += 1
        if keys[pygame.K_w]:
            self.moveDir.y -= 1
        if keys[pygame.K_s]:
            self.moveDir.y += 1
        if self.moveDir.x != 0 or self.moveDir.y != 0:
            self.moveDir = self.moveDir.normalize()
        self.x += self.moveDir.x * self.playerspeed
        self.y += self.moveDir.y * self.playerspeed
    def ShootMouse(self):
        #Player Shooting 
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] and self.shotTime + 1 < time.time():
            click_pos = pygame.mouse.get_pos()
            moveVec = pygame.math.Vector2(click_pos[0] - self.x, click_pos[1] - self.y)
            if moveVec.x != 0 or moveVec.y != 0:
                moveVec = moveVec.normalize()
            bullet = Bullet(self.x, self.y, moveVec, self)
            self.bullets.append(bullet)            
            self.shotTime = time.time()

        for bullet in self.bullets:
            bullet.moveBullet()

    def Move(self, movex : float , movey : float):
        #Player Movement
        self.moveDir.x = movex
        self.moveDir.y = movey

        if self.moveDir.x != 0 or self.moveDir.y != 0:
            self.moveDir = self.moveDir.normalize()
        self.x += self.moveDir.x * self.playerspeed
        self.y += self.moveDir.y * self.playerspeed

    def Shoot(self, boolshoot, shootx, shooty):
        #Player Shooting 
        if boolshoot > 0:
            #moveVec = pygame.math.Vector2(shootx - self.x, shooty - self.y)
            moveVec = pygame.math.Vector2(shootx ,shooty)
            if moveVec.x != 0 or moveVec.y != 0:
                moveVec = moveVec.normalize()
            bullet = Bullet(self.x, self.y, moveVec, self)
            self.bullets.append(bullet)            
            #self.shotTime = time.time()

    def findNearestBullet(self) :
        minDistance = 100000
        if len(self.enemy.bullets) == 0:
            return 0,0,0,0
        for bullet in self.enemy.bullets:
            if bullet.collider is not None:
                distance = math.sqrt((self.x - bullet.x)**2 + (self.y - bullet.y)**2)
                if distance < minDistance:
                    minDistance = distance
                    self.nearestBullet = bullet
        return self.nearestBullet.x, self.nearestBullet.y, self.nearestBullet.direction.x, self.nearestBullet.direction.y
    
    def setEnemy(self, enemy):
        self.enemy = enemy

    def reset(self):
        self.x = random.randint(100,400)
        self.y = random.randint(100,400)
        if self.playerNumber == 1:
            self.x += 500
            self.y += 500

        for bullet in self.bullets :
            bullet.destroyBullet()
        self.bullets.clear()

    def checkCollisionWithEnemyBullet(self) -> bool:
        for bullet in self.enemy.bullets:
            if bullet.collider is not None:
                if self.collider.colliderect(bullet.collider) :
                    bullet.destroyBullet()
                    self.killPlayer()
                    return True
                else: 
                    return False
                

    def killPlayer(self) -> bool:
        print("Player " + str(self.playerNumber) + " " + self.colour +  " died in the Border")
        return True
    

    def draw(self, screen):
        self.collider = pygame.draw.rect(screen, self.colour, (self.x, self.y, self.width, self.height))
        

class Bullet:
    x = 0
    y = 0
    width = 10
    height = 10
    colour = (44, 44, 44)
    shootDuration = 100
    shootSpeed = 8
    TimeToLive = 1000

    def __init__(self, x, y, direction : pygame.math.Vector2, shotBy : Agent):
        self.x = x
        self.y = y
        self.direction = direction
        self.shotby = shotBy
        self.collider = None

    def moveBullet(self):
        self.x += self.direction.x * self.shootSpeed
        self.y += self.direction.y * self.shootSpeed
        self.TimeToLive -= 1
        
        if self.TimeToLive <= 0:
            self.destroyBullet()

    def destroyBullet(self):
        self.shotby.bullets.remove(self)
        del self

    def draw(self, screen):
        self.collider = pygame.draw.rect(screen, self.colour, (self.x, self.y, self.width, self.height))

    def collisionWithPlayer(self) -> bool:
        if self.shotby.enemy.collider is not None:
            if self.collider.colliderect(self.shotby.enemy.collider) :
                return True
            else: 
                return False

class Border: 
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.collider = None
        self.broadth = 10

    def draw(self, screen):
        self.collider = pygame.draw.rect(screen, (255, 0, 0), (self.x, self.y, self.width, self.height), self.broadth)

    def checkCollision(self, player):
        if player is not None:
            if player.collider is not None:
                if not player.collider.colliderect(pygame.Rect(self.broadth +self.x + player.width ,self.broadth + self.y +player.height ,1000 - player.width *2 -(1000- self.width) - self.broadth *2, 1000  - player.height *2 - (1000 -self.height) - self.broadth * 2)):
                    player.killPlayer()
                    return True
        return False

    def decreaseRadius(self, amount):
        self.width -= amount
        self.height -= amount
        self.x += amount / 2
        self.y += amount / 2
        if self.width <= 0 or self.height <= 0:
            self.width = 0
            self.height = 0

