import pygame
import time
from Player import Player
import numpy
from const import width
from GameCompontents import Border

pygame.init()


class Game:
    

    def __init__(self):
        self.clock = pygame.time.Clock()
        self.height = width
        self.width = width
        self.timeUnit = 0
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.shootCooldown = [0, 0]

        self.players = []
        self.squares = []
        self.border = Border(0, 0, self.width, self.height)
        p1 = Player("darkgreen", 0)
        p2 = Player("darkgreen", 1)
        self.addPlayers(p1,p2)

    def addPlayers(self, player1: Player, player2: Player):
        self.players.append(player1)
        self.players.append(player2)
        player1.setEnemy(player2)
        player2.setEnemy(player1)
        player1.playernumber = 0
        player2.playernumber = 1

        self.squares.append(player1.controll)
        self.squares.append(player2.controll)
        self.squares[0].playernumber = 0
        self.squares[1].playernumber = 1

        self.squares[0].reset()
        self.squares[1].reset()

    def playStep(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        done = False
        reward = [0, 0]
        for i in range(len(self.squares)):
            # Move player
            playerAction = action[i]
            self.squares[i].MoveDirection(playerAction)

            # Shoot and bullet movement
            for bullet in self.squares[i].bullets:
                bullet.moveBullet()
                if bullet.collisionWithPlayer(self.squares[i - 1]):
                    print(
                        "Player ",
                        self.squares[i - 1].playernumber + 1,
                        self.squares[i].colour,
                        " got hit by enemy bullet",
                    )
                    reward[i] = 100
                    reward[i - 1] = -100
                    done = True

            if self.shootCooldown[i] > 60:
                self.squares[i].ShootDirection(playerAction)
                self.shootCooldown[i] = 0
            else:
                self.shootCooldown[i] += 1

            # Border collision
            if self.timeUnit == 9:
                self.border.decreaseRadius(2)
            if self.border.checkCollision(self.squares[i]):
                reward[i] = -100
                done = True
            self.timeUnit += 1
            self.timeUnit %= 10

            if reward[i] == 0:
                reward[i] = 1

        self.render()
        self.clock.tick(300)
        # print("FPS: ", self.clock.get_fps())

        return self.getState(), reward, done

    def playStep_New(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        done = False
        reward = [0, 0]
        for i in range(len(self.squares)):
            # Move player
            playerAction = action[i]
            self.squares[i].MoveDirection(playerAction)

            # Shoot and bullet movement
            for bullet in self.squares[i].bullets:
                bullet.moveBullet()
                if bullet.collisionWithPlayer(self.squares[i - 1]):
                    print(
                        "Player ",
                        self.squares[i - 1].playernumber + 1,
                        self.squares[i].colour,
                        " got hit by enemy bullet",
                    )
                    reward[i] = 100
                    reward[i - 1] = -100
                    done = True

            if self.shootCooldown[i] > 60:
                self.squares[i].ShootDirection(playerAction)
                self.shootCooldown[i] = 0
            else:
                self.shootCooldown[i] += 1

            # Border collision
            if self.timeUnit == 9:
                self.border.decreaseRadius(2)
            if self.border.checkCollision(self.squares[i]):
                reward[i] = -100
                done = True
            self.timeUnit += 1
            self.timeUnit %= 10

            if reward[i] == 0:
                reward[i] = 1

        self.render()
        self.clock.tick(300)
        # print("FPS: ", self.clock.get_fps())

        return self.getState(), reward, done

    def getState(self):
        # p0Nx, p0Ny, p0Ndx, p0Ndy = self.players[0].findNearestBullet()
        # p0x, p0y, p0Mx, p0My = self.players[0].getState()

        # p1Nx, p1Ny, p1Ndx, p1Ndy = self.players[1].findNearestBullet()
        # p1x, p1y, p1Mx, p1My = self.players[1].getState()

        nearestBulletp0 = self.squares[0].nearestEnemyBullet

        if nearestBulletp0 is not None:
            p0Nx, p0Ny, p0Ndx, p0Ndy = (
                nearestBulletp0.x,
                nearestBulletp0.y,
                nearestBulletp0.direction.x,
                nearestBulletp0.direction.y,
            )
        else:
            p0Nx, p0Ny, p0Ndx, p0Ndy = 0, 0, 0, 0

        p0x, p0y, p0Mx, p0My = (
            self.squares[0].x,
            self.squares[0].y,
            self.squares[0].moveDir.x,
            self.squares[0].moveDir.y,
        )

        nearestBulletp1 = self.squares[1].nearestEnemyBullet

        if nearestBulletp1 is not None:
            p1Nx, p1Ny, p1Ndx, p1Ndy = (
                nearestBulletp1.x,
                nearestBulletp1.y,
                nearestBulletp1.direction.x,
                nearestBulletp1.direction.y,
            )
        else:
            p1Nx, p1Ny, p1Ndx, p1Ndy = 0, 0, 0, 0

        p1x, p1y, p1Mx, p1My = (
            self.squares[1].x,
            self.squares[1].y,
            self.squares[1].moveDir.x,
            self.squares[1].moveDir.y,
        )
        return numpy.array(
            [
                p0x,
                p0y,
                p0Mx,
                p0My,
                p0Nx,
                p0Ny,
                p0Ndx,
                p0Ndy,
                p1x,
                p1y,
                p1Mx,
                p1My,
                p1Nx,
                p1Ny,
                p1Ndx,
                p1Ndy,
                self.border.width,
            ]
        )

    def reset(self):
        self.squares[0].reset()
        self.squares[1].reset()
        self.border = Border(0, 0, self.width, self.height)
        self.render()
        self.timeUnit = 0
        return self.getState()

    def render(self):
        #   draw to Screen
        self.screen.fill((134, 126, 255))
        for square in self.squares:
            square.draw(self.screen)
            for bullet in square.bullets:
                bullet.draw(self.screen)
        self.border.draw(self.screen)

        self.surface = pygame.surfarray.array3d(pygame.display.get_surface())
        self.height, self.width, self.channels = self.surface.shape
        pygame.display.flip()
