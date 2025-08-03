import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('Arial', 25)
big_font = pygame.font.SysFont('Arial', 48)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 60
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 155, 0)
RED = (200, 0, 0)
DARK_RED = (150, 0, 0)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)

class SnakeGame:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('🐍 Snake RL Game')
        self.clock = pygame.time.Clock()
        self.reset()
        self.show_start_screen()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = Point(x, y)
            if food not in self.snake:
                break
        self.food = food

    def show_start_screen(self):
        self.display.fill(BLACK)
        title = big_font.render("🐍 Snake RL Game", True, WHITE)
        prompt = font.render("Press SPACE to start or ESC to quit", True, WHITE)
        self.display.blit(title, (self.w // 2 - title.get_width() // 2, self.h // 3))
        self.display.blit(prompt, (self.w // 2 - prompt.get_width() // 2, self.h // 2))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit(); quit()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.pause()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def pause(self):
        paused = True
        pause_text = font.render("Paused. Press 'P' to resume.", True, WHITE)
        self.display.blit(pause_text, (self.w // 2 - pause_text.get_width() // 2, self.h // 2))
        pygame.display.flip()
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); quit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    paused = False

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]  # head

        # Wall collision
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True

        # Body collision
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GREY, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GREY, (0, y), (self.w, y))

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, DARK_RED, pygame.Rect(self.food.x+4, self.food.y+4, 12, 12))

        # Draw score
        score_text = font.render(f"Score: {self.score} | Press P to Pause", True, WHITE)
        self.display.blit(score_text, (10, 10))
        pygame.display.flip()

    def _move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]      # straight
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]  # right turn
        else:  # [0, 0, 1]
            new_dir = clockwise[(idx - 1) % 4]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
