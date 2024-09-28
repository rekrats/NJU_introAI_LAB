import os
from PIL import Image
import datetime
import imageio

class BaitEnv:
    def __init__(self, level, render):
        self.level = level
        self.render = render
        self.level2map = {
            0: [
                'wwwww',
                'wgAww',
                'ww..w',
                'w.11w',
                'wwk.w',
                'wwwww',
            ],
            1: [
                'wwwwwwwwwwwww',
                'wwwwwwgwwwwww',
                'wwwww...wwwww',
                'w...w.A.w...w',
                'w.1.......1.w',
                'wwwww.0.wwwww',
                'wwwwww0wwwwww',
                'wwwwwwkwwwwww',
                'wwwwwwwwwwwww',
            ],
            2: [
                'wwwwwwwwwwwww',
                'w...00.00...w',
                'w.w100k001w.w',
                'w.w.00000.w.w',
                'w.1.00m00.1.w',
                'w.w.ww1ww.w.w',
                'w...........w',
                'w.wwww1wwww.w',
                'w.....Ag....w',
                'wwwwwwwwwwwww',
            ],
            3: [
                'wwwwwwwwwwwww',
                'wA....10001gw',
                'w.111110001.w',
                'w1100000001.w',
                'w0001111111.w',
                'w1111.......w',
                'w......11111w',
                'w11111110001w',
                'wm0000000000w',
                'w0000000010kw',
                'wwwwwwwwwwwww',
            ],
            4: [
                'wwwwwww',
                'wkwwwww',
                'w000..w',
                'w0m01.w',
                'w0111.w',
                'w.1A1.w',
                'w01.1.w',
                'wwwwg.w',
                'wwwwwww',
            ],
        }
        self.image_paths = {
            'floor': 'materials/backLBrown.png',
            'hole': 'materials/hole4.png',
            'avatar_nokey': 'materials/swordman1_0.png',
            'avatar_withkey': 'materials/swordmankey1_0.png',
            'mushroom': 'materials/mushroom2.png',
            'key': 'materials/key2.png',
            'goal': 'materials/doorclosed1.png',
            'box': 'materials/block3.png',
            'wall': 'materials/dirtWall_0.png',
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
        # Parse the map and populate the grid
        for y, row in enumerate(self.map):
            for x, char in enumerate(row):
                cell = []
                if char == 'w':
                    cell.append('wall')
                if char == 'g':
                    cell.append('goal')
                    cell.append('floor')
                if char == 'A':
                    cell.append('avatar_nokey')
                    cell.append('floor')
                    self.avatar_pos = (x, y)
                if char == '1':
                    cell.append('box')
                    cell.append('floor')
                if char == 'k':
                    cell.append('key')
                    cell.append('floor')
                if char == '.':
                    cell.append('floor')
                if char == '0':
                    cell.append('hole')
                if char == 'm':
                    cell.append('mushroom')
                    cell.append('floor')
                # By default, add floor if not wall
                if 'floor' not in cell and 'wall' not in cell:
                    cell.append('floor')
                self.grid[y][x] = cell
        self.done = False
        self.score = 0
        self.current_step = 0
        self.goal_exists = True
        if self.render:
            self.do_render()
        return self._get_observation()
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        x, y = self.avatar_pos
        if action == 0:
            dx, dy = 0, 0
        elif action == 4:  # Up
            dx, dy = 0, -1
        elif action == 3:  # Down
            dx, dy = 0, 1
        elif action == 1:  # Left
            dx, dy = -1, 0
        elif action == 2:  # Right
            dx, dy = 1, 0
        else:
            # Invalid action
            return self._get_observation(), 0, self.done, {'error': 'Invalid action'}
        
        new_x, new_y = x + dx, y + dy
        
        # Check bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            # Cannot move outside the grid
            return self._get_observation(), 0, self.done, {'message': 'Hit boundary'}
        
        # Get the entities in the new cell
        new_cell = self.grid[new_y][new_x]
        current_cell = self.grid[y][x]
        reward = 0
        info = {}
        
        # Get the current avatar type
        if 'avatar_nokey' in current_cell:
            avatar_type = 'avatar_nokey'
        elif 'avatar_withkey' in current_cell:
            avatar_type = 'avatar_withkey'
        else:
            # Should not happen
            raise Exception('Avatar not found in current cell')
        
        # Interactions
        # 1. avatar wall > stepBack
        if 'wall' in new_cell:
            info['message'] = 'Hit wall'
            # Do not move
        # 2. avatar hole > killSprite
        elif 'hole' in new_cell:
            current_cell.remove(avatar_type)
            self.avatar_pos = None
            self.done = True
            reward = 0
            info['message'] = 'Fell into hole. Game over.'
        # 3. box avatar  > bounceForward
        elif 'box' in new_cell:
            # Attempt to push box
            box_new_x = new_x + dx
            box_new_y = new_y + dy
            if not (0 <= box_new_x < self.width and 0 <= box_new_y < self.height):
                # Cannot push box outside grid
                info['message'] = 'Cannot push box outside grid'
                # Do not move
            else:
                box_new_cell = self.grid[box_new_y][box_new_x]
                # box wall box mushroom > undoAll
                if 'wall' in box_new_cell or 'box' in box_new_cell or 'mushroom' in box_new_cell:
                    info['message'] = 'Cannot push box. Obstacle ahead.'
                    # Do not move
                else:
                    # Move box
                    self.grid[box_new_y][box_new_x].append('box')
                    self.grid[new_y][new_x].remove('box')
                    # Check if box falls into hole
                    if 'hole' in box_new_cell:
                        self.grid[box_new_y][box_new_x].remove('box')
                        self.grid[box_new_y][box_new_x].remove('hole')
                        reward += 1
                        info['message'] = 'Box fell into hole.'
                    # Check if box is on goal
                    elif 'goal' in box_new_cell:
                        # self.done = True
                        # info['message'] = 'Box pushed onto goal. Game over.'
                        info['message'] = 'Box pushed onto goal.'
                    # Move avatar
                    current_cell.remove(avatar_type)
                    new_cell.append(avatar_type)
                    self.avatar_pos = (new_x, new_y)
        # 6. nokey key > transformTo stype=withkey
        elif 'key' in new_cell:
            if avatar_type == 'avatar_nokey':
                # Transform to withkey
                current_cell.remove(avatar_type)
                new_cell.remove('key')
                new_cell.append('avatar_withkey')
                self.avatar_pos = (new_x, new_y)
                info['message'] = 'Picked up key'
            else:
                # Just move
                current_cell.remove(avatar_type)
                new_cell.remove('key')
                new_cell.append(avatar_type)
                self.avatar_pos = (new_x, new_y)
        # 8. nokey goal > stepBack
        elif 'goal' in new_cell:
            if avatar_type == 'avatar_nokey':
                info['message'] = 'Need key to open goal'
                # Do not move
            else:
                # 9. goal withkey > killSprite scoreChange=5
                new_cell.remove('goal')
                # Move avatar
                current_cell.remove(avatar_type)
                new_cell.append(avatar_type)
                self.avatar_pos = (new_x, new_y)
                reward += 5
                info['message'] = 'Opened goal. You win!'
                # Check termination condition
                goal_exists = any('goal' in cell for row in self.grid for cell in row)
                if not goal_exists:
                    self.done = True
                    self.goal_exists = False
                    info['message'] = 'All goals completed. You win!'
        # 10. mushroom avatar > eat mushroom and increase score
        elif 'mushroom' in new_cell:
            # 增加分数
            reward += 1
            info['message'] = 'Ate a mushroom.'
            # 移除蘑菇
            new_cell.remove('mushroom')
            # 移动avatar
            current_cell.remove(avatar_type)
            new_cell.append(avatar_type)
            self.avatar_pos = (new_x, new_y)
        else:
            # Normal move
            current_cell.remove(avatar_type)
            new_cell.append(avatar_type)
            self.avatar_pos = (new_x, new_y)
        self.current_step += 1
        observation = self._get_observation()
        if self.render:
            self.do_render()
        return observation, reward, self.done, info

    def do_render(self):

        # Load images and cache them
        image_cache = {}
        for key, path in self.image_paths.items():
            if os.path.exists(path):
                image_cache[key] = Image.open(path).convert('RGBA')
            else:
                raise FileNotFoundError(f"Image path {path} does not exist.")

        # Assuming all images are the same size
        tile_width, tile_height = image_cache['floor'].size

        # Create a new image for the entire grid
        grid_width = self.width * tile_width
        grid_height = self.height * tile_height
        grid_image = Image.new('RGBA', (grid_width, grid_height))

        # Define rendering order (from background to foreground)
        layer_order = ['wall', 'goal', 'key', 'box', 'mushroom', 'avatar_nokey', 'avatar_withkey']

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                tile_image = Image.new('RGBA', (tile_width, tile_height))

                # Handle 'hole' separately as it might replace 'floor'
                if 'hole' in cell:
                    tile_image.paste(image_cache['hole'], (0, 0))
                else:
                    tile_image.paste(image_cache['floor'], (0, 0))

                # Paste other elements based on the rendering order
                for layer in layer_order:
                    if layer in cell:
                        tile_image.paste(image_cache[layer], (0, 0), image_cache[layer])

                # Paste the tile onto the grid image
                grid_image.paste(tile_image, (x * tile_width, y * tile_height))

        # Save the final image
        os.makedirs(f'figs/level-{self.level}_{self.timing}/', exist_ok=True)
        grid_image.save(f'figs/level-{self.level}_{self.timing}/step_{self.current_step}.png')
    
    @property
    def action_space(self):
        # Return the available actions
        # 0: Up, 1: Down, 2: Left, 3: Right
        return [0, 1, 2, 3, 4]
    
    def _get_observation(self):
        # For simplicity, return the grid as a list of lists of strings
        return [[list(cell) for cell in row] for row in self.grid]

    def make_gif(self, output_filename="result.gif", duration=0.2):
        image_folder = f'figs/level-{self.level}_{self.timing}/'
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")], 
                        key=lambda x: int(x.split("_")[1].split(".")[0]))
        frames = [imageio.v2.imread(os.path.join(image_folder, img)) for img in images]
        gif_path = os.path.join(image_folder, output_filename)
        imageio.mimsave(gif_path, frames, duration=duration)
