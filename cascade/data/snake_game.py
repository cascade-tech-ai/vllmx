import pygame
import random
import sys

# Constants
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
FPS = 10

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 180, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.body = [(5, 5), (4, 5), (3, 5)]
        self.direction = RIGHT

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)

    def change_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def collides_with_self(self):
        return self.body[0] in self.body[1:]

    def collides_with_wall(self):
        x, y = self.body[0]
        return not (0 <= x < WIDTH // CELL_SIZE and 0 <= y < HEIGHT // CELL_SIZE)

class Food:
    def __init__(self):
        self.position = self.random_position()

    def random_position(self):
        return (random.randint(0, WIDTH // CELL_SIZE - 1),
                random.randint(0, HEIGHT // CELL_SIZE - 1))

    def draw(self, surface):
        rect = pygame.Rect(self.position[0] * CELL_SIZE,
                           self.position[1] * CELL_SIZE,
                           CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, RED, rect)

def draw_grid(surface):
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(surface, DARK_GREEN, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, DARK_GREEN, (0, y), (WIDTH, y))

def draw_snake(surface, snake):
    for segment in snake.body:
        rect = pygame.Rect(segment[0] * CELL_SIZE,
                           segment[1] * CELL_SIZE,
                           CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GREEN, rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    snake = Snake()
    food = Food()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: snake.change_direction(UP)
        elif keys[pygame.K_DOWN]: snake.change_direction(DOWN)
        elif keys[pygame.K_LEFT]: snake.change_direction(LEFT)
        elif keys[pygame.K_RIGHT]: snake.change_direction(RIGHT)

        snake.move()

        if snake.collides_with_self() or snake.collides_with_wall():
            break

        if snake.body[0] == food.position:
            snake.grow()
            food = Food()

        screen.fill(BLACK)
        draw_grid(screen)
        draw_snake(screen, snake)
        food.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
