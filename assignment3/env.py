import os
from PIL import Image
import datetime
import random

class AliensEnv:
    def __init__(self, level, render):
        self.level = level
        self.render = render
        self.level2map = {
            0: [
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
                'w                              w',
                'w1                             w',
                'w000                           w',
                'w000                           w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w    000      000000     000   w',
                'w   00000    00000000   00000  w',
                'w   0   0    00    00   00000  w',
                'w                A             w',
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
            ],
            1: [
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
                'w                              w',
                'w2                             w',
                'w000                           w',
                'w000                           w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w    000      000000     000   w',
                'w   00000    00000000   00000  w',
                'w   0   0    00    00   00000  w',
                'w                A             w',
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
            ],
            2: [
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
                'w                              w',
                'w1                             w',
                'w000                           w',
                'w000                           w',
                'w                              w',
                'w        0000        0000      w',
                'w        0  0        0  0      w',
                'w                              w',
                'w                              w',
                'w   00000    00000000   00000  w',
                'w   0   0    00    00   00000  w',
                'w                A             w',
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
            ],
            3: [
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
                'w                              w',
                'w1                             w',
                'w000                           w',
                'w000                           w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                             Aw',
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
            ],
            4: [
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
                'w                              w',
                'w1                             w',
                'w000                           w',
                'w000                           w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w                              w',
                'w000000000000000000000000000000w',
                'w                              w',
                'w                              w',
                'wA                             w',
                'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',
            ],
        }
        self.char_mapping = {
            'w': ['wall'],
            '0': ['base'],
            '1': ['portalSlow'],
            '2': ['portalFast'],
            'A': ['avatar'],
            ' ': [],
        }
        self.image_paths = {
            'wall': 'materials/wall.png',
            'base': 'materials/base.png',
            'avatar': 'materials/avatar.png',
            'portalSlow': 'materials/portal.png',
            'portalFast': 'materials/portal.png',
            'alien': 'materials/alien.png',
            'sam': 'materials/spaceship.png',
            'bomb': 'materials/bomb.png',
            'floor': 'materials/floor.png',
        }
        for k, v in self.image_paths.items():
            assert os.path.exists(v), (k, v)

        self.reset()

    def reset(self):
        self.timing = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.map = self.level2map[self.level]
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.grid = [[[] for _ in range(self.width)] for _ in range(self.height)]
        self.avatar_pos = None
        self.current_step = 0
        self.done = False
        self.score = 0
        self.info = {}
        for y, row in enumerate(self.map):
            for x, char in enumerate(row):
                cell = []
                if char in self.char_mapping:
                    cell.extend(self.char_mapping[char])
                    if 'avatar' in cell:
                        self.avatar_pos = (x, y)
                self.grid[y][x] = cell
        self.portal_cooldowns = {}
        self.portal_totals = {}
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if 'portalSlow' in cell:
                    self.portal_cooldowns[(x, y)] = 16
                    self.portal_totals[(x, y)] = 20
                elif 'portalFast' in cell:
                    self.portal_cooldowns[(x, y)] = 12
                    self.portal_totals[(x, y)] = 20
        self.aliens = []
        self.alien_directions = {}
        self.alien_cooldowns = {}
        self.bombs = []
        if self.render:
            self.do_render()
        return self._get_observation()

    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, self.info
        reward = 0
        info = {}
        x, y = self.avatar_pos
        if action == 0:
            pass
        elif action == 1:  # Left
            dx, dy = -1, 0
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                if 'wall' not in self.grid[new_y][new_x]:
                    # Move avatar
                    self.grid[y][x].remove('avatar')
                    self.grid[new_y][new_x].append('avatar')
                    self.avatar_pos = (new_x, new_y)
        elif action == 2:  # Right
            dx, dy = 1, 0
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                if 'wall' not in self.grid[new_y][new_x]:
                    # Move avatar
                    self.grid[y][x].remove('avatar')
                    self.grid[new_y][new_x].append('avatar')
                    self.avatar_pos = (new_x, new_y)
        elif action == 3:  # Shoot
            sam_exists = False
            for row in self.grid:
                for cell in row:
                    if 'sam' in cell:
                        sam_exists = True
                        break
                if sam_exists:
                    break
            if not sam_exists:
                self.grid[y][x].append('sam')
        else:
            info['error'] = 'Invalid action'

        self.move_aliens()
        self.move_bombs()
        self.move_sams()
        self.spawn_aliens()
        reward += self.handle_interactions()
        self.check_termination()
        if self.render:
            self.do_render()
        observation = self._get_observation()
        self.current_step += 1
        return observation, reward, self.done, self.info

    def move_aliens(self):
        new_aliens = []
        new_alien_directions = {}
        for alien in self.aliens:
            x, y = alien['pos']
            direction = alien['direction']
            bomb_cooldown = alien['bomb_cooldown']
            dx = direction
            dy = 0
            new_x = x + dx
            new_y = y + dy
            if self.is_valid_position(new_x, new_y):
                if 'wall' not in self.grid[new_y][new_x]:
                    self.grid[y][x].remove('alien')
                    self.grid[new_y][new_x].append('alien')
                    alien['pos'] = (new_x, new_y)
                    x, y = new_x, new_y
                else:
                    direction = -direction
                    alien['direction'] = direction
            else:
                direction = -direction
                alien['direction'] = direction
            if bomb_cooldown > 0:
                alien['bomb_cooldown'] -= 1
            else:
                if random.random() < 0.01:
                    bomb_x, bomb_y = alien['pos']
                    self.grid[bomb_y][bomb_x].append('bomb')
                    self.bombs.append({'pos': (bomb_x, bomb_y)})
                    alien['bomb_cooldown'] = 3
            new_aliens.append(alien)
            new_alien_directions[(x, y)] = direction
        self.aliens = new_aliens
        self.alien_directions = new_alien_directions

    def move_bombs(self):
        new_bombs = []
        for bomb in self.bombs:
            x, y = bomb['pos']
            dx, dy = 0, 1
            new_x, new_y = x + dx, y + dy
            self.grid[y][x].remove('bomb')
            if self.is_valid_position(new_x, new_y):
                if 'wall' not in self.grid[new_y][new_x]:
                    self.grid[new_y][new_x].append('bomb')
                    bomb['pos'] = (new_x, new_y)
                    new_bombs.append(bomb)
                else:
                    pass
            else:
                pass
        self.bombs = new_bombs

    def move_sams(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if 'sam' in cell:
                    dx, dy = 0, -1
                    new_x, new_y = x + dx, y + dy
                    cell.remove('sam')
                    if self.is_valid_position(new_x, new_y):
                        if 'wall' not in self.grid[new_y][new_x]:
                            self.grid[new_y][new_x].append('sam')
                        else:
                            pass
                    else:
                        pass

    def spawn_aliens(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if 'portalSlow' in cell or 'portalFast' in cell:
                    portal_type = 'portalSlow' if 'portalSlow' in cell else 'portalFast'
                    cooldown = self.portal_cooldowns.get((x, y), 0)
                    total = self.portal_totals.get((x, y), 0)
                    if total <= 0:
                        cell.remove(portal_type)
                        continue
                    if cooldown > 0:
                        self.portal_cooldowns[(x, y)] -= 1
                    else:
                        self.grid[y][x].append('alien')
                        self.aliens.append({'pos': (x, y), 'direction': -1, 'bomb_cooldown': 3})
                        self.alien_directions[(x, y)] = -1
                        if portal_type == 'portalSlow':
                            self.portal_cooldowns[(x, y)] = 16
                        else:
                            self.portal_cooldowns[(x, y)] = 12
                        self.portal_totals[(x, y)] -= 1

    def handle_interactions(self):
        reward = 0
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if 'base' in cell and 'bomb' in cell:
                    cell.remove('base')
                    cell.remove('bomb')
                    self.remove_bomb_at((x, y))
                if 'base' in cell and 'sam' in cell:
                    cell.remove('base')
                    cell.remove('sam')
                    reward += 1
                if 'base' in cell and 'alien' in cell:
                    cell.remove('alien')
                    self.remove_alien_at((x, y))
                if 'avatar' in cell and 'alien' in cell:
                    cell.remove('avatar')
                    self.done = True
                    reward -= 1
                    self.info['message'] = 'Hit by alien. You lose.'
                if 'avatar' in cell and 'bomb' in cell:
                    cell.remove('avatar')
                    self.done = True
                    reward -= 1
                    self.info['message'] = 'Hit by bomb. You lose.'
                if 'alien' in cell and 'sam' in cell:
                    cell.remove('alien')
                    cell.remove('sam')
                    reward += 2
                    self.remove_alien_at((x, y))
        self.score += reward
        return reward

    def remove_bomb_at(self, pos):
        x, y = pos
        for bomb in self.bombs:
            if bomb['pos'] == pos:
                self.bombs.remove(bomb)
                break

    def remove_alien_at(self, pos):
        for alien in self.aliens:
            if alien['pos'] == pos:
                self.aliens.remove(alien)
                break

    def check_termination(self):
        avatar_exists = False
        portal_exists = False
        alien_exists = False
        for row in self.grid:
            for cell in row:
                if 'avatar' in cell:
                    avatar_exists = True
                if 'portalSlow' in cell or 'portalFast' in cell:
                    portal_exists = True
                if 'alien' in cell:
                    alien_exists = True
        if not avatar_exists:
            self.done = True
            self.info['message'] = 'Avatar destroyed. You lose.'
        elif not portal_exists and not alien_exists:
            self.done = True
            self.info['message'] = 'All aliens and portals destroyed. You win.'

    @property
    def action_space(self):
        return [0, 1, 2, 3]

    def _get_observation(self):
        return [[list(cell) for cell in row] for row in self.grid]

    def do_render(self):
        image_cache = {}
        for key, path in self.image_paths.items():
            if os.path.exists(path):
                image_cache[key] = Image.open(path).convert('RGBA')
            else:
                raise FileNotFoundError(f"Image path {path} does not exist.")

        tile_width, tile_height = image_cache['wall'].size

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

        os.makedirs(f'figs/aliens_{self.timing}/', exist_ok=True)
        grid_image.save(f'figs/aliens_{self.timing}/step_{self.current_step}.png')
