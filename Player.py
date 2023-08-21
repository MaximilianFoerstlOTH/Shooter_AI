from GameCompontents import Square


class Player:
    def __init__(self, color, playernumber) -> None:
        self.playernumber = playernumber
        self.color = color

        self.controll = Square(color)
        self.controll.reset()
        self.enemy = None

    def getAction():
        pass

    def getBullets(self) -> []:
        return self.controll.bullets

    def setEnemy(self, enemy):
        self.enemy = enemy
