import os
import sys
import pygame
from PIL import Image

from env import BaitEnv

class BaitEnvPygame(BaitEnv):
    def do_render(self):
        # 与原始的 do_render 方法相同，但返回图像而不是保存
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

        layer_order = ['wall', 'goal', 'key', 'box', 'mushroom', 'avatar_nokey', 'avatar_withkey']

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                tile_image = Image.new('RGBA', (tile_width, tile_height))

                if 'hole' in cell:
                    tile_image.paste(image_cache['hole'], (0, 0))
                else:
                    tile_image.paste(image_cache['floor'], (0, 0))

                for layer in layer_order:
                    if layer in cell:
                        tile_image.paste(image_cache[layer], (0, 0), image_cache[layer])

                grid_image.paste(tile_image, (x * tile_width, y * tile_height))

        return grid_image

def main():
    # 初始化 Pygame
    pygame.init()

    # 创建环境实例
    env = BaitEnvPygame(level=1, render=False)

    # 获取初始渲染的图像
    grid_image = env.do_render()

    # 将 PIL 图像转换为 Pygame 表面
    mode = grid_image.mode
    size = grid_image.size
    data = grid_image.tobytes()
    pygame_image = pygame.image.fromstring(data, size, mode)

    # 设置显示窗口
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Bait Game')

    # 显示初始图像
    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    # 游戏循环
    done = False
    step = 0
    clock = pygame.time.Clock()
    while not done:
        # 控制帧率
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                    break

                # 将箭头键映射到动作
                elif event.key == pygame.K_UP:
                    action = 4  # 上
                elif event.key == pygame.K_DOWN:
                    action = 3  # 下
                elif event.key == pygame.K_LEFT:
                    action = 1  # 左
                elif event.key == pygame.K_RIGHT:
                    action = 2  # 右
                else:
                    action = 0  # 无操作

                observation, reward, game_over, info = env.step(action)
                print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")
                step += 1

                # 重新渲染图像
                grid_image = env.do_render()
                mode = grid_image.mode
                size = grid_image.size
                data = grid_image.tobytes()
                pygame_image = pygame.image.fromstring(data, size, mode)

                # 更新显示
                screen.blit(pygame_image, (0, 0))
                pygame.display.flip()

                if game_over:
                    print("游戏结束!")
                    print(f"信息: {info}")
                    done = True
                    break

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
