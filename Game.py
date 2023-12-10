from Agent import *


def whether_exit(event):
    if event.type == pygame.QUIT:
        print("游戏退出...")
        pygame.quit()
        # exit()


class Game:
    def __init__(self, window_width, window_height) -> None:
        super().__init__()
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))

        self.bg = pygame.image.load(r'.\images\background.png')
        self.bird_images = [pygame.image.load(fr'.\images\bird{i}.png') for i in range(3)]
        self.down_tube_image = pygame.image.load(fr'.\images\tube.png')
        self.up_tube_image = pygame.transform.flip(self.down_tube_image, False, True)
        self.ground_image = pygame.image.load(fr'.\images\ground.png')
        self.number_images = [pygame.image.load(fr'.\images\{i}.png') for i in range(10)]
        self.GG_image = pygame.image.load(fr'.\images\GameOver.png')
        self.message_image = pygame.image.load(fr'.\images\message.png')

        self.ground = Ground(self.ground_image, -(self.ground_image.get_width() - window_width),
                             window_height - self.ground_image.get_height(),
                             window_width)

        self.clock = pygame.time.Clock()

        self.tubes = None
        self.game_over_flag = None
        self.score = None
        self.flash_flag = None
        self.myBird = None
        self.myBirds = []
        self.BirdBrains = []
        self.myBirds_index = []
        self.fitness = []
        self.optimization_flag = False
        self.partial_init()

    def partial_init(self):
        self.myBird = BirdSprite(self.bird_images, round(self.window_width * 0.2),
                                 (self.window_height - self.bird_images[0].get_height()) // 2)
        up_tube, down_tube = make_random_tubes(self.up_tube_image, self.down_tube_image, self.window_width,
                                               self.window_height,
                                               True)
        self.tubes = [[up_tube, down_tube]]
        self.game_over_flag = False
        self.score = 0
        self.flash_flag = True

    def flash_run(self):
        for event in pygame.event.get():
            whether_exit(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.myBird.reset_play_mode()
                self.flash_flag = False
        self.myBird.update()
        self.ground.update()
        self.myBird.draw(self.screen)
        self.ground.draw(self.screen)
        draw_message(self.message_image, self.screen, self.window_width, self.window_height)

    def play_run(self, action=None, actions: list = None):
        # events
        for event in pygame.event.get():
            whether_exit(event)
            if action is None and not self.optimization_flag:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.myBird.flap_flag = True

        # action
        if not self.optimization_flag:
            if action is not None:
                self.myBird.flap_flag = (True if action == 1 else False)
            self.myBird.update()
        else:
            for i in range(len(actions)):
                self.myBirds[self.myBirds_index[i]].flap_flag = (True if actions[i] == 1 else False)
                self.myBirds[self.myBirds_index[i]].update()

        # whether gen new tubes
        if self.window_width - self.tubes[-1][0].rect.x - self.tubes[-1][0].rect.w > self.tubes[-1][0].rect.w * 2.5:
            up_tube, down_tube = make_random_tubes(self.up_tube_image, self.down_tube_image, self.window_width,
                                                   self.window_height)
            self.tubes.append([up_tube, down_tube])

        # whether drop tubes
        if self.tubes[0][0].rect.x + self.tubes[0][0].rect.w < 0:
            self.tubes.pop(0)

        # ground update
        self.ground.update()

        # ground's reason: whether game over/ whether remove birds and draw birds
        if not self.optimization_flag:
            self.game_over_flag = self.game_over_flag or self.myBird.rect.colliderect(self.ground.rect)
            self.myBird.draw(self.screen)
        else:
            tmp_birds_index = deepcopy(self.myBirds_index)
            for i in tmp_birds_index:
                if self.myBirds[i].rect.colliderect(self.ground.rect):
                    self.myBirds_index.remove(i)
                    self.fitness[i] = self.myBirds[i].frame
                else:
                    self.myBirds[i].draw(self.screen)

        for i in range(len(self.tubes)):

            # tubes' reason: whether game over/ whether remove birds
            if not self.optimization_flag:
                self.game_over_flag = self.game_over_flag or self.tubes[i][0].rect.colliderect(self.myBird.rect) or \
                                      self.tubes[i][1].rect.colliderect(self.myBird.rect)
            else:
                tmp_birds_index = deepcopy(self.myBirds_index)
                for j in tmp_birds_index:
                    if self.tubes[i][0].rect.colliderect(self.myBirds[j].rect) or self.tubes[i][1].rect.colliderect(
                            self.myBirds[j].rect):
                        self.myBirds_index.remove(j)
                        self.fitness[j] = self.myBirds[j].frame

            # tubes update
            self.tubes[i][0].update()
            self.tubes[i][1].update()

            # choose next tube and/or add score
            if not self.optimization_flag:
                if self.tubes[i][0].score_flag is False and self.tubes[i][0].rect.x + self.tubes[i][
                    0].rect.w < self.myBird.rect.x:
                    self.score += 1
                    self.tubes[i][0].score_flag = True
            else:
                if len(self.myBirds_index) > 0:
                    if self.tubes[i][0].score_flag is False and self.tubes[i][0].rect.x + self.tubes[i][0].rect.w < \
                            self.myBirds[self.myBirds_index[0]].rect.x:
                        self.score += 1
                        self.tubes[i][0].score_flag = True

            # draw tubes
            self.tubes[i][0].draw(self.screen)
            self.tubes[i][1].draw(self.screen)

        self.ground.draw(self.screen)
        draw_score(self.number_images, self.score, self.screen, self.window_width, self.window_height)

    def dead_run(self):
        self.myBird.mode = 1
        for event in pygame.event.get():
            whether_exit(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.partial_init()
        if self.myBird.rect.y + self.myBird.rect.h < self.window_height - self.ground.rect.h:
            self.myBird.update()
        for i in range(len(self.tubes)):
            self.tubes[i][0].draw(self.screen)
            self.tubes[i][1].draw(self.screen)
        self.ground.draw(self.screen)
        self.myBird.draw(self.screen)
        draw_score(self.number_images, self.score, self.screen, self.window_width, self.window_height)
        draw_game_over(self.GG_image, self.screen, self.window_width, self.window_height)

    def run(self):
        while True:
            self.clock.tick(30)
            self.screen.blit(self.bg, (0, 0))
            if self.flash_flag:
                self.flash_run()
            else:
                if not self.game_over_flag:
                    self.play_run()
                else:
                    self.dead_run()
            pygame.display.update()

    def ai_run(self, action, epoch=None):
        self.clock.tick(30)
        self.screen.blit(self.bg, (0, 0))
        self.play_run(action)
        if epoch is not None:
            draw_epoch(self.number_images, epoch, self.screen, self.window_width, self.window_height)
        pygame.display.update()
        return calculate_state(self.myBird, self.tubes)

    def optimization_init(self, n, m=10, weights: list = None):
        self.optimization_flag = True
        self.myBirds = []
        self.BirdBrains = []
        for i in range(n):
            tmp_bird = BirdSprite(self.bird_images, round(self.window_width * 0.2),
                                  (self.window_height - self.bird_images[0].get_height()) // 2)
            tmp_bird.reset_play_mode()
            self.myBirds.append(tmp_bird)
            if weights is None:
                self.BirdBrains.append(BirdBrain(m))
            else:
                tmp_brain = BirdBrain(m)
                tmp_brain.input_weights(weights[i])
                self.BirdBrains.append(tmp_brain)
        self.myBirds_index = [i for i in range(n)]
        self.fitness = [0 for _ in range(n)]

    def optimization_run(self, epoch):
        while True:
            self.clock.tick(30)
            self.screen.blit(self.bg, (0, 0))
            actions = []
            for i in range(len(self.myBirds_index)):
                dx, dy = calculate_state(self.myBirds[self.myBirds_index[i]], self.tubes)
                actions.append(self.BirdBrains[self.myBirds_index[i]].forward(np.array([dx, dy])))
            if len(actions) == 0:
                break
            self.play_run(actions=actions)
            if epoch is not None:
                draw_epoch(self.number_images, epoch, self.screen, self.window_width, self.window_height)
            pygame.display.update()
        weights = []
        for i in range(len(self.BirdBrains)):
            weights.append(self.BirdBrains[i].output_weights())
        return np.array(self.fitness), weights


if __name__ == "__main__":
    game = Game(288, 512)
    game.run()
