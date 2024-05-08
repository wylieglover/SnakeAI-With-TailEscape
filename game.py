import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from astar import find_path

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 30

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.path = None
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.path = None
        self.model_moves = 0
        self.engine_moves = 0

    def _place_food(self):
        min_distance = 2 * BLOCK_SIZE  # Minimum distance from snake
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food_position = Point(x, y)

            # Check distance from snake
            too_close = any(abs(food_position.x - part.x) + abs(food_position.y - part.y) < min_distance
                            for part in self.snake)
            
            if not too_close:
                self.food = food_position
                break

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        head_x, head_y = self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE
        food_x, food_y = self.food.x // BLOCK_SIZE, self.food.y // BLOCK_SIZE
      
        path = find_path((head_x, head_y), (food_x, food_y), 
                [(p.x // BLOCK_SIZE, p.y // BLOCK_SIZE) for p in self.snake],
                self.w // BLOCK_SIZE, self.h // BLOCK_SIZE)
        
        print("New path: ", path)
        
        reward = 0
        game_over = False  
        
        if path:
            self.path = path
            for coords in path[1:]:
                x = round(coords[0] * BLOCK_SIZE)
                y = round(coords[1] * BLOCK_SIZE)

                self._move(x, y, action=None)
                self.snake.insert(0, self.head)

                if self.is_collision() or self.frame_iteration > 100*len(self.snake):
                    game_over = True
                    reward = -10
                    print("Game Over!")
                    return reward, game_over, self.score

                # 4. place new food or just move
                if self.head == self.food:
                    self.score += 1
                    reward = 10
                    self._place_food()
                else:
                    self.snake.pop()
                        
                # 5. update ui and clocks
                self._update_ui()
                self.clock.tick(SPEED)
        else:
            self._move(self.head.x, self.head.y, action)
            self.snake.insert(0, self.head)
                      
            if self.is_collision() or self.frame_iteration > 100*len(self.snake):
                game_over = True
                reward = -10
                print("Game Over!")
                return reward, game_over, self.score

            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()    
            
            # 5. update ui and clocks
            self._update_ui()
            self.clock.tick(SPEED)  
            
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, x, y, action): 
        # print("Model's action: ", action) # <--- Spams console
        if action is not None:
            print("Model's action: ", action)
            self.model_moves += 1
            # print(self.model_moves) # <--- Spams console
            # [straight, right, left]
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx] # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
            else: # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

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
        else:
            self.engine_moves += 1
            # print(self.engine_moves)  # <--- Spams console
            dx = x - self.head.x
            dy = y - self.head.y

            # Determine the new direction based on the change in coordinates
            if dx > 0:
                self.direction = Direction.RIGHT
            elif dx < 0:
                self.direction = Direction.LEFT
            elif dy > 0:
                self.direction = Direction.DOWN
            elif dy < 0:
                self.direction = Direction.UP
        
        self.head = Point(x, y)
        