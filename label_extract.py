# 以下為從phh文件提取部分關鍵資訊的程式，不見得適用於所有情況，不保證適用任何phh文件，建議自己寫一支更全面的特徵提取程式

import ast
import os

def parse_phh(file_path):
    phh_info = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("antes ="):
            phh_info['antes'] = ast.literal_eval(line.split("=")[1].strip())
        elif line.startswith("blinds_or_straddles ="):
            phh_info['blinds_or_straddles'] = ast.literal_eval(line.split("=")[1].strip())
        elif line.startswith("min_bet ="):
            phh_info['min_bet'] = int(line.split("=")[1].strip())
        elif line.startswith("players ="):
            phh_info['players'] = ast.literal_eval(line.split("=")[1].strip())
        elif line.startswith("actions ="):
            actions = ast.literal_eval(line.split("=")[1].strip())
            phh_info['actions'] = actions
            phh_info['deck'] = [action.split(" ")[-1] for action in actions if action.startswith('d dh')]
        elif line.startswith("starting_stacks ="):
            phh_info['starting_stacks'] = ast.literal_eval(line.split("=")[1].strip())
        elif line.startswith("finishing_stacks ="):
            phh_info['finishing_stacks'] = ast.literal_eval(line.split("=")[1].strip())
    phh_info['ori_n_players'] = len(phh_info['starting_stacks'])
    return phh_info

def parse_deck_code(deck):
    assert type(deck) == str
    assert len(deck) % 2 == 0
    lv_map = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    color_map = ['s','h','d','c']
    codes = []
    for i in range(len(deck)//2):
        lv=deck[i*2]
        color=deck[i*2+1]
        for id, l in enumerate(lv_map):
            if l == lv:
                lid = id
                break
        for id, c in enumerate(color_map):
            if c == color:
                cid = id
                break
        codes.append([lid,cid])
    return codes

def parse_deck(phh_info):
    decks = [parse_deck_code(deck) for deck in phh_info['deck']]
    decks1 = [(x[0][0] + x[0][1]*13 + 1) for x in decks]
    decks2 = [(x[1][0] + x[1][1]*13 + 1) for x in decks]
    phh_info['deck1'] = decks1
    phh_info['deck2'] = decks2
    return phh_info

def parse_seat(phh_info):
    n_player = phh_info['ori_n_players']
    phh_info['seat'] = list(range(1, n_player + 1))
    return phh_info

def parse_play(phh_info):
    n_player = phh_info['ori_n_players']
    first_act = phh_info['actions'][n_player:n_player*2]
    play = []
    for i in range(n_player):
        if f"p{i+1} f" in first_act:
            play.append(0)
        else:
            play.append(1)
    phh_info['play'] = play
    return phh_info

def parse_chip(phh_info):
    n_player = phh_info['ori_n_players']
    start_chip = phh_info['starting_stacks']
    end_chip = phh_info['finishing_stacks']
    win_or_lose_chip_amount = [(start - end) for (start, end) in zip(start_chip,end_chip)]
    win_or_lose_chip_condition = [(1 if (x > 0) else (-1 if (x < 0) else 0)) for x in win_or_lose_chip_amount]
    bankrupt = [1 if (end == 0) else 0 for end in end_chip]
    phh_info['win_or_lose_chip_amount'] = win_or_lose_chip_amount
    phh_info['win_or_lose_chip_condition'] = win_or_lose_chip_condition
    phh_info['bankrupt'] = bankrupt
    return phh_info

def player_pad(phh_info,to_n_player = None):
    if to_n_player is None:
        return phh_info
    
    all_key = list(phh_info.keys())
    pad = to_n_player - phh_info['ori_n_players']
    for i in all_key:
        if isinstance(phh_info[i],list):
            if len(phh_info[i]) == phh_info['ori_n_players']:
                if isinstance(phh_info[i][0],str):
                    phh_info[i] += pad*['']
                else:
                    phh_info[i] += pad*[0]
    phh_info['padto_n_players'] = to_n_player
    return phh_info

def label_extract(filename, n_player = None, extract_filter = None):
    phh_info = parse_phh(filename) # ori_n_players antes blinds_or_straddles min_bet players actions deck starting_stacks finishing_stacks
    phh_info = parse_deck(phh_info) # +deck1 deck2
    phh_info = parse_seat(phh_info) # +seat
    phh_info = parse_play(phh_info) # +play
    phh_info = parse_chip(phh_info) # +win_or_lose_chip_amount win_or_lose_chip_condition bankrupt
    phh_info = player_pad(phh_info,n_player) # +padto_n_players (fix len of feature)
    if extract_filter is not None:
        phh_info_ex = dict([])
        for key in extract_filter:
            phh_info_ex[key] = phh_info[key]
        return phh_info_ex
    return phh_info

def to_single(phh_info, return_key = False):
    keys = list(phh_info.keys())
    n_players = len(phh_info[keys[0]])
    single_info = []
    for i in range(n_players):
        if phh_info['starting_stacks'][i] == 0:
            continue
        info = []
        for key in keys:
            info.append(phh_info[key][i])
        single_info.append(info)
    if return_key:
        return single_info, keys
    return single_info

def extract_from_folder(phh_dir, max_players = 9):
    interest_list = ['antes','blinds_or_straddles','starting_stacks','deck1','deck2','seat','play','finishing_stacks','win_or_lose_chip_amount','win_or_lose_chip_condition','bankrupt']
    file_paths = []
    for file_name in os.listdir(phh_dir):
        if file_name.endswith(".phh"):
            file_path = os.path.join(phh_dir, file_name)
            file_paths.append(file_path)
    phh_infos = []
    single_infos = []
    for file_path in file_paths:
        phh_info = label_extract(file_path,max_players,interest_list)
        single_info = to_single(phh_info)
        phh_infos.append(phh_info)
        single_infos += single_info
    return phh_infos, single_infos, interest_list

if __name__ == '__main__':
    # interest_list = ['antes','blinds_or_straddles','starting_stacks','finishing_stacks','deck1','deck2','seat']
    # phh_info = label_extract("hand_record/1.phh",9,interest_list)
    # print(phh_info)
    # single_info = to_single(phh_info)
    # for i in single_info:
    #     print(i)
    phh_info, single_info, interest_list = extract_from_folder("hand_record")
    print(phh_info)
    for i in single_info:
        print(i)
    print(interest_list)