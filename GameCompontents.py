import pygame
import time
import numpy
from const import width
import random
import math


class Square:
    width = 50
    height = 50
    playerspeed = 5

    def __init__(self, colour) -> None:
        self.colour = colour
        self.playernumber = 0
        self.x = 0
        self.y = 0
        self.moveDir = pygame.math.Vector2(0, 0)
        self.bullets = []
        self.collider = None
        self.enemy = None
        self.moveDir = pygame.math.Vector2(0, 0)
        self.nearestEnemyBullet = None

    def draw(self, screen):
        self.collider = pygame.draw.rect(
            screen, self.colour, (self.x, self.y, self.width, self.height)
        )

    def killPlayer(self) -> bool:
        print(
            "Player "
            + str(self.playernumber)
            + " "
            + self.colour
            + " died in the Border"
        )
        return True

    def checkCollisionWithEnemyBullet(self) -> bool:
        for bullet in self.enemy.bullets:
            if bullet.collider is not None:
                if self.collider.colliderect(bullet.collider):
                    bullet.destroyBullet()
                    self.killPlayer()
                    return True
                else:
                    return False

    def MoveWitKey(self):
        # Player Movement
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
        # Player Shooting
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

    def Move(self, movex: float, movey: float):
        # Player Movement
        self.moveDir.x = movex
        self.moveDir.y = movey

        if self.moveDir.x != 0 or self.moveDir.y != 0:
            self.moveDir = self.moveDir.normalize()
        self.x += self.moveDir.x * self.playerspeed
        self.y += self.moveDir.y * self.playerspeed

    def MoveDirection(self, direction):
        ## 1 = left, 2 = right, 3 = up, 4 = down
        if direction == 0:
            self.moveDir.x = -1
            self.moveDir.y = 0
        elif direction == 1:
            self.moveDir.x = 1
            self.moveDir.y = 0
        elif direction == 2:
            self.moveDir.x = 0
            self.moveDir.y = -1
        elif direction == 3:
            self.moveDir.x = 0
            self.moveDir.y = 1
        else:
            self.moveDir.x = 0
            self.moveDir.y = 0
        self.x += self.moveDir.x * self.playerspeed
        self.y += self.moveDir.y * self.playerspeed

    def Shoot(self, boolshoot, shootx, shooty):
        # Player Shooting
        if boolshoot > 0:
            # moveVec = pygame.math.Vector2(shootx - self.x, shooty - self.y)
            moveVec = pygame.math.Vector2(shootx, shooty)
            if moveVec.x != 0 or moveVec.y != 0:
                moveVec = moveVec.normalize()
            bullet = Bullet(self.x, self.y, moveVec, self)
            self.bullets.append(bullet)

    def ShootDirection(self, direction):
        ## 4 = shootleft, 5 = shootright, 6 = shootup, 7 = shootdown
        if direction == 4:
            moveVec = pygame.math.Vector2(-1, 0)
        elif direction == 5:
            moveVec = pygame.math.Vector2(1, 0)
        elif direction == 6:
            moveVec = pygame.math.Vector2(0, -1)
        elif direction == 7:
            moveVec = pygame.math.Vector2(0, 1)
        else:
            return
        bullet = Bullet(self.x, self.y, moveVec, self)
        self.bullets.append(bullet)

    def reset(self):
        x = random.randint(width / 10, width / 2 - width / 10)
        y = random.randint(width / 10, width / 2 - width / 10)

        if self.playernumber == 1:
            x += width / 2
            y += width / 2

        self.x = x
        self.y = y
        for bullet in self.bullets:
            bullet.destroyBullet()
        self.bullets.clear()

    def findNearestEnemyBullet(self, enemy):
        if enemy is None:
            return
        nearestBullet = None
        nearestDistance = 10000
        for bullet in enemy.bullets:
            if bullet.collider is not None:
                distance = math.sqrt(
                    (bullet.x - self.x) ** 2 + (bullet.y - self.y) ** 2
                )
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearestBullet = bullet
        self.nearestEnemyBullet = nearestBullet


class Bullet:
    x = 0
    y = 0
    width = 10
    height = 10
    colour = (44, 44, 44)
    shootDuration = 100
    shootSpeed = 8
    TimeToLive = 1000

    def __init__(self, x, y, direction: pygame.math.Vector2, shotBy):
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
        self.collider = pygame.draw.rect(
            screen, self.colour, (self.x, self.y, self.width, self.height)
        )

    def collisionWithPlayer(self, player) -> bool:
        if player == self.shotby:
            return False
        if player.collider is not None:
            if self.collider.colliderect(player.collider):
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
        self.collider = pygame.draw.rect(
            screen, (255, 0, 0), (self.x, self.y, self.width, self.height), self.broadth
        )

    def checkCollision(self, player):
        if player is not None:
            if player.collider is not None:
                if not player.collider.colliderect(
                    pygame.Rect(
                        self.broadth + self.x + player.width,
                        self.broadth + self.y + player.height,
                        1000
                        - player.width * 2
                        - (1000 - self.width)
                        - self.broadth * 2,
                        1000
                        - player.height * 2
                        - (1000 - self.height)
                        - self.broadth * 2,
                    )
                ):
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
