import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import heapq

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
RED = (200, 0, 0)
BLACK = (0, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 186, 0)
EYE_COLOR = (255, 255, 255)
PUPIL_COLOR = (0, 0, 0)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
BROWN = (139, 69, 19)

BLOCK_SIZE = 20
SPEED = 100

class SnakeGameAI:
    def __init__(self, w=400, h=400):
        if w % BLOCK_SIZE != 0 or h % BLOCK_SIZE != 0:
            raise ValueError("Width and Height must be multiples of BLOCK_SIZE.")
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2*BLOCK_SIZE, self.head.y)]
        self.food = None
        self._place_food() 
        self.score = 0
        self.frame_iteration = 0
        self.engine_moves = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward = 0
        game_over = False  
        
        self._move()
        self.snake.insert(0, self.head)
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            print("Game Over!")
            return reward, game_over, self.score
        
        # 4. place new food or just move
        if self.head == self.food:  
            self.score += 1
            # check for cover all
            if self.score == (self.w // BLOCK_SIZE * self.h // BLOCK_SIZE) - 3:
                game_over = True
                print("!!Game Won!!")
                return reward, game_over, self.score
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
        snake_length = len(self.snake)
        
        for i, pt in enumerate(self.snake):
            if i == 0:  # Draw head with googly eyes
                pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # Draw eyes
                eye_radius = 3
                pupil_radius = 2
                # Left eye
                eye_x = pt.x + 5
                eye_y = pt.y + 5
                pygame.draw.circle(self.display, EYE_COLOR, (eye_x, eye_y), eye_radius)
                pygame.draw.circle(self.display, PUPIL_COLOR, (eye_x, eye_y), pupil_radius)
                # Right eye
                eye_x = pt.x + BLOCK_SIZE - 5
                pygame.draw.circle(self.display, EYE_COLOR, (eye_x, eye_y), eye_radius)
                pygame.draw.circle(self.display, PUPIL_COLOR, (eye_x, eye_y), pupil_radius)
            elif i == snake_length - 1:  # Draw tail with rattle design
                pygame.draw.rect(self.display, BROWN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:  # Draw the body with gradient effect
                ratio = i / snake_length
                body_color = self.interpolate_color(DARK_GREEN, LIGHT_GREEN, ratio)
                pygame.draw.rect(self.display, body_color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    @staticmethod
    def interpolate_color(start_color, end_color, ratio):
        return (
            int(start_color[0] + (end_color[0] - start_color[0]) * ratio),
            int(start_color[1] + (end_color[1] - start_color[1]) * ratio),
            int(start_color[2] + (end_color[2] - start_color[2]) * ratio),
        )

    def _move(self): 
        head = [self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE]
        food = [self.food.x // BLOCK_SIZE, self.food.y // BLOCK_SIZE]
        snake = [[pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE] for pt in self.snake]
        
        x_length = self.w // BLOCK_SIZE
        y_length = self.h // BLOCK_SIZE
        x_max = x_length - 1
        y_max = y_length - 1

        next_move = None

        # Priority queue to store cells based on their heuristic values
        priority_queue = []

        # Evaluate heuristic for each adjacent cell
        for move in adjacencies(head, x_max, y_max):
            if move not in snake:
                h_value = heuristic(move, x_length, y_length, snake, food)
                # Push cell and its heuristic value onto the priority queue
                heapq.heappush(priority_queue, (h_value, move))

        # Extract the move with the best heuristic (lowest value in a min-heap)
        while priority_queue:
            _, move = heapq.heappop(priority_queue)
            potential_path = search(head, move, x_max, y_max, snake)
            if potential_path and is_path_safe(snake, potential_path, x_max, y_max):
                next_move = move
                break

        # If no heuristic-based move found, revert to previous logic
        if not next_move:
            possible_moves = adjacencies(head, x_max, y_max)
            safe_moves = [move for move in possible_moves if move not in snake and is_path_safe(snake, [move], x_max, y_max)]
            if safe_moves:
                next_move = safe_moves[0]

        # If still no move found, try to continue in the previous direction
        if not next_move:
            next_move = [
                self.head.x + BLOCK_SIZE if self.direction == Direction.RIGHT else self.head.x - BLOCK_SIZE if self.direction == Direction.LEFT else self.head.x,
                self.head.y + BLOCK_SIZE if self.direction == Direction.DOWN else self.head.y - BLOCK_SIZE if self.direction == Direction.UP else self.head.y
            ]

        next_move[0] = max(0, min(next_move[0], x_max))
        next_move[1] = max(0, min(next_move[1], y_max))

        old_x, old_y = head
        new_x, new_y = next_move

        if new_x == old_x + 1:
            self.direction = Direction.RIGHT
        elif new_x == old_x - 1:
            self.direction = Direction.LEFT
        elif new_y == old_y - 1:
            self.direction = Direction.UP
        elif new_y == old_y + 1:
            self.direction = Direction.DOWN

        # Update head position based on new direction
        if self.direction == Direction.RIGHT:
            self.head = Point(self.head.x + BLOCK_SIZE, self.head.y)
        elif self.direction == Direction.LEFT:
            self.head = Point(self.head.x - BLOCK_SIZE, self.head.y)
        elif self.direction == Direction.DOWN:
            self.head = Point(self.head.x, self.head.y + BLOCK_SIZE)
        elif self.direction == Direction.UP:
            self.head = Point(self.head.x, self.head.y - BLOCK_SIZE)
        
        self.engine_moves += 1
        
# FUNCTIONS FOR SNAKE ESCAPE ALGORITHM        
def heuristic(cell, x_length, y_length, snake, point):
    size = x_length * y_length * 2
    x_max = x_length - 1
    y_max = y_length - 1

    if not includes(adjacencies(snake[0], x_max, y_max), cell):
        return 0

    path_to_point = search(cell, point, x_max, y_max, snake)
    if path_to_point:
        snake_at_point = shift(snake, path_to_point, True)
        for next_cell in difference(adjacencies(point, x_max, y_max), snake_at_point):
            if search(next_cell, tail(snake_at_point), x_max, y_max, snake_at_point):
                return len(path_to_point)

    path_to_tail = search(cell, tail(snake), x_max, y_max, snake)
    if path_to_tail:
        return size - len(path_to_tail)

    return size * 2

def search(start, end, x_max, y_max, snake):
    queue = deque([start])
    paths = {tuple(start): [start]}
    visited = set()

    while queue:
        current = queue.popleft()
        if tuple(current) in visited:
            continue
        visited.add(tuple(current))

        snake_shifted = shift(snake, paths[tuple(current)], False)
        
        if equals(current, end):
            return paths[tuple(current)]
        
        for next_cell in difference(adjacencies(current, x_max, y_max), snake_shifted):
            if tuple(next_cell) not in paths:
                queue.append(next_cell)
                paths[tuple(next_cell)] = paths[tuple(current)] + [next_cell]

    return None  # Return None to handle no path found scenario

def adjacencies(a, x_max, y_max):
    potential = [
        [a[0], a[1] - 1],  # up
        [a[0] + 1, a[1]],  # right
        [a[0], a[1] + 1],  # down
        [a[0] - 1, a[1]]  # left
    ]
    return [coord for coord in potential if 0 <= coord[0] <= x_max and 0 <= coord[1] <= y_max]

def equals(a, b):
    return a[0] == b[0] and a[1] == b[1]

def includes(a, b):
    return any(equals(x, b) for x in a)

def difference(a, b):
    b_set = set(map(tuple, b))  # Convert to set for faster lookup
    return [x for x in a if tuple(x) not in b_set]

def shift(snake, path, collect=False):
    result = path + snake
    return result[:len(path) + len(snake) - len(path) + (1 if collect else 0)]

def tail(snake):
    return snake[-1]

def is_path_safe(snake, path, x_max, y_max):
    if not path:
        return False
    temp_snake = shift(snake, path, True)  # simulate the snake's new position
    return search(path[-1], tail(temp_snake), x_max, y_max, temp_snake) is not None
    
