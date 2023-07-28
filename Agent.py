import pygame
import time
from DQLearning import DQN
from torch.autograd import Variable
import torch
import random

class Agent:
    x = 0
    y = 0
    width = 50
    height = 50
    playerspeed = 5

    def __init__(self, x, y, colour, playernumber):
        self.colour = colour
        self.x = x
        self.y = y
        self.bullets = []
        self.collider = None
        self.enemy = None
        self.moveDir = pygame.math.Vector2(0, 0)
        self.shotTime = time.time()
        self.inputs = []
        self.NN = DQN().cuda()

    def getAction(self, frame, reward, done):
        #input = Variable(torch.Tensor([frame for _ in range(1000*1000)])).cuda()
        frameTensor = torch.Tensor(frame)
        inputTensor = frameTensor.view(1, 3, 1000, 1000)
        inputTensor = Variable(inputTensor).cuda()
        self.inputs = self.NN.forward(inputTensor)
        self.Move(random.random(), random.random(), random.random(), random.random())
        self.Shoot(random.random(), random.random(), random.random())
        #self.Move(self.inputs[0][0], self.inputs[0][1], self.inputs[0][2], self.inputs[0][3])
        #self.Shoot(self.inputs[0][4], self.inputs[0][5], self.inputs[0][6])

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

    def Move(self, l, r, u, d):
        #Player Movement
        self.moveDir = pygame.math.Vector2(0, 0)
        keys = pygame.key.get_pressed()
        if l > 0.8:
            self.moveDir.x -= 1
        if r > 0.8:
            self.moveDir.x += 1
        if u > 0.8:
            self.moveDir.y -= 1
        if d > 0.8:
            self.moveDir.y += 1
        if self.moveDir.x != 0 or self.moveDir.y != 0:
            self.moveDir = self.moveDir.normalize()
        self.x += self.moveDir.x * self.playerspeed
        self.y += self.moveDir.y * self.playerspeed

    def Shoot(self, boolshoot, shootx, shooty):
        #Player Shooting 
        if boolshoot > 0.8 and self.shotTime + 1 < time.time():
            #moveVec = pygame.math.Vector2(shootx - self.x, shooty - self.y)
            moveVec = pygame.math.Vector2(shootx ,shooty)
            if moveVec.x != 0 or moveVec.y != 0:
                moveVec = moveVec.normalize()
            bullet = Bullet(self.x, self.y, moveVec, self)
            self.bullets.append(bullet)            
            self.shotTime = time.time()

        for bullet in self.bullets:
            bullet.moveBullet()
    def setEnemy(self, enemy):
        self.enemy = enemy


    def checkCollisionWithEnemyBullet(self) -> bool:
        for bullet in self.enemy.bullets:
            if bullet.collider is not None:
                if self.collider.colliderect(bullet.collider) :
                    bullet.destroyBullet()
                    return self.killPlayer()
                else: 
                    return False
                

    def killPlayer(self) -> bool:
        print("Player died")
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

    def checkCollision(self, player) -> int:
        if player is not None:
            if player.collider is not None:
                if not player.collider.colliderect(pygame.Rect(self.broadth +self.x + player.width ,self.broadth + self.y +player.height ,1000 - player.width *2 -(1000- self.width) - self.broadth *2, 1000  - player.height *2 - (1000 -self.height) - self.broadth * 2)):
                    return player.killPlayer()
                
                else:
                    return 0
    def decreaseRadius(self, amount):
        self.width -= amount
        self.height -= amount
        self.x += amount / 2
        self.y += amount / 2
        if self.width <= 0 or self.height <= 0:
            self.width = 0
            self.height = 0
