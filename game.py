import pygame 
import time

class Game:
    characters = []
    def __init__(self):
        self.player = Character(400,300, "blue")
        self.enemy = Character(700,700, "red")
        self.player.setEnemy(self.enemy)
        self.enemy.setEnemy(self.player)
        self.characters.append(self.player)
        self.characters.append(self.enemy)
        self.startGame()

    def playStep(self):
        for character in self.characters:
            character.Move()
            character.Shoot()
            character.checkCollisionWithEnemyBullet()

    def startGame(self):
        pygame.init()
        screen = pygame.display.set_mode((1000, 1000))
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            #draw to Screen
            screen.fill((255, 255, 255))
            for character in self.characters:
                character.draw(screen)
                for bullet in character.bullets:
                    bullet.draw(screen)

            #Play step
            self.playStep()


            # flip() the display to put your work on screen
            pygame.display.flip()
            #Set FPS
            clock.tick(60)

        pygame.quit()


class Character:
    x = 0
    y = 0
    width = 50
    height = 50
    playerspeed = 5

    def __init__(self, x, y, colour):
        self.colour = colour
        self.x = x
        self.y = y
        self.bullets = []
        self.collider = None
        self.enemy = None
        self.moveDir = pygame.math.Vector2(0, 0)
        self.shotTime = time.time()


    def Move(self):
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

    def Shoot(self):
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

    def setEnemy(self, enemy):
        self.enemy = enemy

    def checkCollisionWithEnemyBullet(self):
        for bullet in self.enemy.bullets:
            if bullet.collider is not None:
                if self.collider.colliderect(bullet.collider) :
                    bullet.destroyBullet()
                    ##TODO add negative reward

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

    def __init__(self, x, y, direction : pygame.math.Vector2, shotBy : Character):
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


game = Game()