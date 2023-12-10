import pygame
import random


class BirdSprite:

    def __init__(self, images, x, y) -> None:
        super().__init__()

        self.image = images[0]
        self.images = images
        self.image_idx = [0, 1, 2, 1]
        self.init_y = y
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.min_y = -self.rect.h

        # test
        self.max_y = 512 - self.rect.h
        # test

        self.acc = 0.5
        self.v_y = 1
        self.max_v_y = 10
        self.rot = 0
        self.v_rot = -3
        self.min_rot = -90
        self.flap_v = -9  # 飞一下直接将速度更新
        self.flap_flag = False
        self.frame = 0
        self.mode = 2  # 0代表正常游戏，1代表碰撞，2代表开场动画
        self.max_flash_v_y = 4
        self.min_flash_v_y = -4

    def change_image(self):
        self.frame += 1
        if self.frame % 5 == 0:
            self.image = self.images[self.image_idx[(self.frame // 5) % 4]]

    def update(self):
        if self.mode == 0:
            self.change_image()
            if self.flap_flag:
                self.v_y = -9
                self.rot = 30
                self.flap_flag = False
            else:
                self.v_y = min(self.v_y + self.acc, self.max_v_y)
            self.rect.y = min(self.max_y, max(self.rect.y + self.v_y, self.min_y))
            self.rot = max(self.v_rot + self.rot, self.min_rot)
        elif self.mode == 1:
            self.image = self.images[1]
            self.v_y += self.acc
            self.rect.y = max(self.rect.y + self.v_y, self.min_y)
            self.rot = max(self.v_rot + self.rot, self.min_rot)
        else:
            self.change_image()
            if self.v_y >= self.max_flash_v_y or self.v_y <= self.min_flash_v_y:
                self.acc *= -1
            self.v_y += self.acc
            self.rect.y += self.v_y

    def reset_play_mode(self):
        self.mode = 0
        self.v_y = -9
        self.acc = 1
        self.rect.y = self.init_y
        self.rot = 30

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.rot)
        rotated_rect = rotated_image.get_rect(center=self.rect.center)
        screen.blit(rotated_image, rotated_rect)


class TubeSprite:

    def __init__(self, image, x, y) -> None:
        super().__init__()

        self.image = image
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.score_flag = False
        self.v_x = -5

    def update(self):
        self.rect.x += self.v_x

    def draw(self, screen):
        screen.blit(self.image, self.rect)


class Ground:

    def __init__(self, image, x, y, window_width) -> None:
        super().__init__()
        self.image = image
        self.rect = pygame.Rect(x, y, self.image.get_width(), self.image.get_height())
        self.v_x = 2
        self.extra_x = window_width - self.rect.w

    def update(self):
        self.rect.x = (self.rect.x + self.v_x) % self.extra_x

    def draw(self, screen):
        screen.blit(self.image, self.rect)


def make_random_tubes(up_image, down_image, window_width, window_height, init_flag=False):
    tube_gap = 120
    tube_height = up_image.get_height()
    if init_flag:
        tube_x = window_width + 30
    else:
        tube_x = window_width + 10
    true_height = window_height * 0.79
    up_y = random.randint(round(true_height * 0.2), round(true_height * 0.8) - tube_gap)

    return TubeSprite(up_image, tube_x, up_y - tube_height), TubeSprite(down_image, tube_x, up_y + tube_gap)


def draw_score(images, score, screen, window_width, window_height):
    number_list = [int(x) for x in str(score)]
    selected_images = []
    for i in range(len(number_list)):
        selected_images.append(images[number_list[i]])
    w = 0
    h = 0
    for i in range(len(selected_images)):
        w += selected_images[i].get_width()
        h = max(h, selected_images[i].get_height())
    x = (window_width - w) // 2
    y = window_height // 10
    for i in range(len(selected_images)):
        if i == 0:
            screen.blit(selected_images[i], pygame.Rect(x, y, selected_images[i].get_width(), h))
        else:
            x += selected_images[i - 1].get_width()
            screen.blit(selected_images[i],
                        pygame.Rect(x, y, selected_images[i].get_width(), h))


def draw_game_over(image, screen, window_width, window_height):
    w = image.get_width()
    h = image.get_height()
    x = (window_width - w) // 2
    y = window_height // 5
    screen.blit(image, pygame.Rect(x, y, w, h))


def draw_message(image, screen, window_width, window_height):
    w = image.get_width()
    h = image.get_height()
    x = (window_width - w) // 2
    y = window_height * 0.12
    screen.blit(image, pygame.Rect(x, y, w, h))
