import os
import sys
import pygame
from PIL import Image
import pickle
from env import AliensEnv

class AliensEnvPygame(AliensEnv):
    def __init__(self, level=0, render=False):
        super().__init__(level=level, render=render)
        self.frames = []
        self.log_folder = f'logs/game_records_lvl{level}_{self.timing}'
        os.makedirs(self.log_folder, exist_ok=True)

    def do_render(self):
        image_cache = {}
        for key, path in self.image_paths.items():
            if os.path.exists(path):
                image_cache[key] = Image.open(path).convert('RGBA')
            else:
                raise FileNotFoundError(f"Image path {path} does not exist.")

        tile_width, tile_height = image_cache['floor'].size
        grid_width = self.width * tile_width
        grid_height = self.height * tile_height
        grid_image = Image.new('RGBA', (grid_width, grid_height))

        layer_order = ['floor', 'base', 'portalSlow', 'portalFast', 'alien', 'bomb', 'sam', 'avatar', 'wall']

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                tile_image = Image.new('RGBA', (tile_width, tile_height))
                tile_image.paste(image_cache['floor'], (0, 0))

                for layer in layer_order:
                    if layer in cell:
                        tile_image.paste(image_cache[layer], (0, 0), image_cache[layer])

                grid_image.paste(tile_image, (x * tile_width, y * tile_height))

        self.frames.append(grid_image.copy())
        return grid_image

    def save_gif(self, filename='replay.gif', duration=0.1):
        frames = [frame.convert('P', palette=Image.ADAPTIVE) for frame in self.frames]
        frames[0].save(os.path.join(self.log_folder, filename),
                       save_all=True,
                       append_images=frames[1:],
                       duration=duration * 1000,
                       loop=0)

def main():
    pygame.init()

    env = AliensEnvPygame(level=0, render=False)

    data = []

    observation = env.reset()

    grid_image = env.do_render()

    mode = grid_image.mode
    size = grid_image.size
    data_image = grid_image.tobytes()
    pygame_image = pygame.image.fromstring(data_image, size, mode)

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Aliens Game')

    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    done = False
    step = 0
    clock = pygame.time.Clock()
    while not done:
        clock.tick(15)

        action = 0  # 默认无操作

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

                elif event.key == pygame.K_LEFT:
                    action = 1  # 左移
                elif event.key == pygame.K_RIGHT:
                    action = 2  # 右移
                elif event.key == pygame.K_UP:
                    action = 3  # 发射
                else:
                    action = 0  # 无操作

                data.append((observation, action))

        observation, reward, game_over, info = env.step(action)
        if reward != 0 or info:
            print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")
        step += 1

        grid_image = env.do_render()
        mode = grid_image.mode
        size = grid_image.size
        data_image = grid_image.tobytes()
        pygame_image = pygame.image.fromstring(data_image, size, mode)

        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

        if game_over:
            print("游戏结束!")
            print(f"信息: {info}")
            done = True

    with open(f'{env.log_folder}/data.pkl', 'wb') as f:
        pickle.dump(data, f)

    env.save_gif()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
