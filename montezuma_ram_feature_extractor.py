import ipdb
import numpy as np
from collections import defaultdict
from utils import _parse_state

from feature_extractor import FeatureExtractor

def _get_index(address):
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col

def get_byte(ram, address):
    idx = _get_index(address)
    return ram[idx]

def bcd2int(bcd_string):
    nibbles = [bcd_string[i:i + 4] for i in range(0, len(bcd_string), 4)]
    digits = [format(int(nib, 2), '01d') for nib in nibbles]
    return int(''.join(digits), 10)

class MonteRAMParser:
    def __init__(self):
        self.reset()
        
        self.status_codes_dict = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead',  # dive 1
            0xC9: 'dead',  # dive 2
            0xC8: 'dead',  # dissolve 1
            0xDD: 'dead',  # dissolve 2
            0xFD: 'dead',  # smoke 1
            0xE7: 'dead',  # smoke 2
        }

        self.object_type_dict = {
            0: 'none',
            1: 'jewel',
            2: 'sword',
            3: 'mallet',
            4: 'key',
            5: 'jump_skull',
            6: 'torch',
            8: 'snake',
            10: 'spider'
        }
        self.object_configuration_dict = {
            0: 'one_single',  # normal object
            1: 'two_near',  # two objects, as close as possible
            2: 'two_mid',  # same positions as three_near with center obj removed
            3: 'three_near',  # same distance apart as two_near
            4: 'two_far',  # same positions as three_mid with center obj removed
            5: 'one_double',  # double-wide object
            6: 'three_mid',  # same distance apart as two_mid
            7: 'one_triple',  # triple-wide object
        }

    def vectorize_ram(self, state):
        def dict_val_to_idx(dictionary, val):
            return list(dictionary.values()).index(val)

        vector = []
        for i, (k, v) in enumerate(state.items()):
            if k in ['player_x', 'player_y', 'skull_x', 'lives']:
                # normalized features should be appended as floats, not ints
                vector.append(v)
            elif k == "player_status":
                vector.append(dict_val_to_idx(self.status_codes_dict, v))
            elif k == "object_type":
                vector.append(dict_val_to_idx(self.object_type_dict, v))
            elif k == "object_configuration":
                vector.append(dict_val_to_idx(self.object_configuration_dict, v))
            elif k == "inventory":
                vector.extend([int(has_item) for _item, has_item in v])
            elif k in ["player_look", "object_dir", "skull_dir"]:
                vector.append(int(v == "left"))
            elif k in ["door_left", "door_right"]:
                vector.append(int(v == "locked"))
            elif k == "object_vertical_dir":
                vector.append(int(v == "up"))
            else:
                assert isinstance(v, (int, np.bool_, np.uint)), 'did not correctly vectorize state'
                vector.append(int(v))

        return vector

    def reset(self):
        self.state = dict()
        self.skull_dir = 'left'
        self.object_vertical_dir = 'up'
        self.lives = 5

    def prune_for_first_room_distance_prediction(self, state):
        pruned_state = dict()
        for k in ['player_x', 'player_y', 'player_look', 'player_jumping', 'player_falling', 'score', 'lives', 'door_left', 'door_right', 'has_skull', 'skull_x', 'skull_dir', 'respawning', 'just_died']:
            # normalize x, y
            if k in ['player_x', 'skull_x']:
                pruned_state[k] = (state[k] - 1) / 151
            elif k == 'player_y':
                pruned_state[k] = (state[k] - 148) / 104
            elif k == 'lives':
                pruned_state[k] = state[k] / 5.
            else:
                pruned_state[k] = state[k]
        pruned_state['has_key'] = state['inventory']['key'] > 0
        pruned_state['on_rope'] = state['player_status'] in ['on-rope', 'climbing-rope']
        pruned_state['on_ladder'] = state['player_status'] in ['on-ladder', 'climbing-ladder']
        return pruned_state
        
    def prune_for_proof_of_concept(self, state):
        pruned_state = dict()
        for k in ['player_x', 'player_y', 'lives', 'has_key', 'on_rope', 'on_ladder']:
                pruned_state[k] = state[k]
        return pruned_state

    def parseRAM(self, ram):
        """Get the current annotated RAM state dictonary”
        See RAM annotations:
        https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
        """
        state = dict()
        state['screen'] = get_byte(ram, '83')
        state['level'] = get_byte(ram, 'b9')
        state['screen_changing'] = get_byte(ram, '84') != 0

        bcd_score = ''.join([format(get_byte(ram, '9' + str(i)), '010b')[2:] for i in [3, 4, 5]])
        state['score'] = bcd2int(bcd_score)

        state['has_ladder'] = state['screen'] not in [5, 8, 12, 14, 15, 16, 17, 18, 20, 23]
        state['has_rope'] = state['screen'] in [1, 5, 8, 14]
        state['has_lasers'] = state['screen'] in [0, 7, 12]
        state['has_platforms'] = state['screen'] == 8
        state['has_bridge'] = state['screen'] in [10, 18, 20, 22]
        state['time_to_appear'] = get_byte(ram, 'd3')
        frame = get_byte(ram, '80')
        state['time_to_disappear'] = -int(frame) % 128 if state['time_to_appear'] == 0 else 0  # yapf: disable

        x = int(get_byte(ram, 'aa'))
        y = int(get_byte(ram, 'ab'))
        state['player_x'] = x
        state['player_y'] = y

        state['player_jumping'] = 1 if get_byte(ram, 'd6') != 0xFF else 0
        state['player_falling'] = 1 if get_byte(ram, 'd8') != 0 else 0
        status = get_byte(ram, '9e')
        state['player_status'] = self.status_codes_dict[status]

        look = int(format(get_byte(ram, 'b4'), '08b')[-4])
        state['player_look'] = 'left' if look == 1 else 'right'

        state['lives'] = get_byte(ram, 'ba')
        if (state['lives'] < self.lives):
            state['just_died'] = 1
        else:
            state['just_died'] = 0
        self.lives = state['lives']

        state['time_to_spawn'] = get_byte(ram, 'b7')
        state['respawning'] = (state['time_to_spawn'] > 0 or state['player_status'] == 'dead')

        ram_inventory = format(get_byte(ram, 'c1'), '08b')  # convert to binary
        possible_items = ['torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer']
        inventory = defaultdict(int)
        for item, bit in zip(possible_items, ram_inventory):
            inventory[item] += int(bit)
        state['inventory'] = inventory

        # yapf: disable
        objects = format(get_byte(ram, 'c2'), '08b')[-4:]  # convert to binary; keep last 4 bits
        state['door_left'] = 'locked' if int(objects[0]) == 1 and state['screen'] in [1, 5, 17] else 'unlocked'
        state['door_right'] = 'locked' if int(objects[1]) == 1 and state['screen'] in [1, 5, 17] else 'unlocked'
        state['has_skull'] = int(objects[2]) if state['screen'] in [1, 5, 18] else 0 # skull screens
        if state['screen'] in [1, 5, 17]: # door screens
            state['has_object'] = int(objects[3])
        else:
            state['has_object'] = sum([int(c) for c in objects])
        # yapf: enable

        object_type = get_byte(ram, 'b1')
        state['object_type'] = self.object_type_dict[object_type]
        object_configuration = int(format(get_byte(ram, 'd4'), '08b')[-3:], 2)  # convert to binary; keep last 3 bits -- yapf: disable
        state['object_configuration'] = self.object_configuration_dict[object_configuration]
        state['has_spider'] = (state['has_object'] and state['object_type'] == 'spider')
        state['has_snake'] = (state['has_object'] and state['object_type'] == 'snake')
        state['has_jump_skull'] = (state['has_object'] and state['object_type'] == 'jump_skull')
        state['has_enemy'] = state['has_spider'] or state['has_snake'] or state['has_jump_skull']
        state['has_jewel'] = (state['has_object'] and state['object_type'] == 'jewel')

        state['object_x'] = int(get_byte(ram, 'ac'))
        state['object_y'] = int(get_byte(ram, 'ad'))
        state['object_y_offset'] = int(get_byte(ram, 'bf'))  # ranges from 0 to f
        obj_direction_bit = int(format(get_byte(ram, 'b0'), '08b')[-4], 2)
        state['object_dir'] = 'right' if obj_direction_bit == 1 else 'left'

        skull_offset = defaultdict(lambda: 33, {
            18: [22,23,12][state['level']],
        })[state['screen']]  # yapf: disable
        state['skull_x'] = int(get_byte(ram, 'af')) + skull_offset
        # Note: up to some rounding, player dies when |player_x - skull_x| <= 6
        if 'skull_x' in self.state.keys():
            if state['skull_x'] - self.state['skull_x'] > 0:
                self.skull_dir = 'right'
            if state['skull_x'] - self.state['skull_x'] < 0:
                self.skull_dir = 'left'
        state['skull_dir'] = self.skull_dir

        if 'object_y' in self.state.keys():
            if state['object_y'] - self.state['object_y'] > 0:
                self.object_vertical_dir = 'up'
            elif state['object_y'] - self.state['object_y'] < 0:
                self.object_vertical_dir = 'down'
        state['object_vertical_dir'] = self.object_vertical_dir

        self.state = state
        return self.vectorize_ram(self.prune_for_proof_of_concept(self.prune_for_first_room_distance_prediction(state)))


class MontezumaRamFeatureExtractor(FeatureExtractor):
    def extract_features(self, traj):
        """
        Extracts features of RAM state for distance learning 
        (masks the frame number RAM bit and adds in extra info 
        like skull direction to make the state Markov)

        traj: List of [RAM state, image state]
        return a list of parsed RAM states
        """
        ram_states = list(zip(*traj))[0]
        image_states = list(zip(*traj))[1]
        parser = MonteRAMParser()
        parsed_states = [parser.parseRAM(s) for s in ram_states]
        return parsed_states, image_states

