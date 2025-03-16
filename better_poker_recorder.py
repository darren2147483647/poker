from pokerkit import *
import re

def read_from_file(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

def card1(seg):
    return bool(re.match(r'^[2-9ATJQKatjqk][hdsc]$', seg, re.IGNORECASE))

def card2(seg):
    return bool(re.match(r'^[2-9ATJQKatjqk][hdsc][2-9ATJQKatjqk][hdsc]$', seg, re.IGNORECASE))

def card3(seg):
    return bool(re.match(r'^[2-9ATJQKatjqk][hdsc][2-9ATJQKatjqk][hdsc][2-9ATJQKatjqk][hdsc]$', seg, re.IGNORECASE))

def even_index_upper(s: str) -> str:
    result = [
        char.upper() if i % 2 == 0 and char.isalpha() else char
        for i, char in enumerate(s)
    ]
    return "".join(result)

def parse_from_utg(segments):
    assert len(segments)%2==0
    n = len(segments)//2
    return segments[n-2:n]+segments[:n-2]+segments[n:]

def parse_deal_hole_first(segments):
    assert len(segments)%2==0
    return segments[::2]+segments[1::2]

def parse_poker_log_pre(log: str):
    pattern = r'([2-9atjqk][hdsc][2-9atjqk][hdsc]|\d+\.5|\d+m|cc|f|\*.*?\*)'
    segments = re.findall(pattern, log, re.IGNORECASE)
    n_segments = []
    f=0
    for seg in segments:
        if card2(seg):
            if f>0:
                n_segments.append("f")
                f-=1
            n_segments.append(even_index_upper(seg))
            f+=1
        elif seg[0]=="*":
            continue
        else:
            n_segments.append(seg)
            f-=1
    if f>0:
        n_segments.append("f")
    segments = n_segments
    segments = parse_deal_hole_first(segments)
    segments = parse_from_utg(segments)
    return segments

def parse_poker_log_other(log: str,merge_flop=False):
    pattern = r'([2-9atjqk][hdsc]|\d+\.5|\d+m|cc|f|\*.*?\*)'
    segments = re.findall(pattern, log, re.IGNORECASE)
    n_segments = []
    for seg in segments:
        if card1(seg):
            n_segments.append(even_index_upper(seg))
        elif seg[0]=="*":
            continue
        else:
            n_segments.append(seg)
    if merge_flop:
        for i,seg in enumerate(n_segments):
            if card1(seg):
                n_seg = n_segments[i]+n_segments[i+1]+n_segments[i+2]
                n_segments = n_segments[:i] + [n_seg,] + n_segments[i+3:]
                merge_flop = False
                break
    segments = n_segments
    return segments, merge_flop

def parse_poker_log_all(log):
    segments=[]
    seg = parse_poker_log_pre(log[0])
    segments=segments+seg
    f=1
    for i in log[1:]:
        seg, f = parse_poker_log_other(i, merge_flop=f)
        segments=segments+seg
    return segments

def check_hh(filename):
    # Load hand
    with open(filename, "rb") as file:
        hh = HandHistory.load(file)
    # Create game
    game = hh.create_game()
    # Create state
    state = hh.create_state()
    # Iterate through each action step
    print(hh.starting_stacks)
    print(hh.finishing_stacks)
    print("===action===")
    for state, action in hh.state_actions:
        print(action)
    print("===final===")
    print(state.stacks)

def fix_finishing(filename,fix_filename):
    # Load hand
    with open(filename, "rb") as file:
        hh = HandHistory.load(file)
    # Create game
    game = hh.create_game()
    # Create state
    state = hh.create_state()
    # Iterate through each action step
    print(state.stacks)
    for state, action in hh.state_actions:
        pass
    print(state.stacks)
    hh.finishing_stacks=state.stacks
    # Dump hand
    with open(fix_filename, "wb") as file:
        hh.dump(file)
        print(f"save record as {fix_filename}")

def read_hh(filename, prepare_final_stack = True):
    # Load hand
    with open(filename, "rb") as file:
        hh = HandHistory.load(file)
    # Create game
    game = hh.create_game()
    # Create state
    state = hh.create_state()
    players = hh.players
    others={"event": hh.event,
            "day": hh.day,
            "month": hh.month,
            "year": hh.year}
    if prepare_final_stack:
        for state, action in hh.state_actions:
            pass

    return game, state, players, others

def write_hh(filename, game, state, players, others = {  "event": '2024 World Series of Poker Paradise Event #9: $25,000 Super Main Event | Day 5 (FINAL TABLE)',
                                                "day": 20,
                                                "month": 12,
                                                "year": 2024}, fix = True):
    # Creating hand history
    hh = HandHistory.from_game_state(game, state)
    hh.players = players
    hh.event = others["event"]
    hh.day = others["day"]
    hh.month = others["month"]
    hh.year = others["year"]
    # Dump hand
    with open(filename, "wb") as file:
        hh.dump(file)
        print(f"save record as {filename}")
    if fix:
        fix_finishing(filename,filename)

def new_game(blind=[1500000,3000000,3000000]):
    # Game state construction
    game = NoLimitTexasHoldem(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        True,
        (0,blind[2]),
        (blind[0], blind[1]),
        blind[1],
    )
    return game

def next_state(game,players,start_stack,rotate=True):
    assert len(players)==len(start_stack)
    if rotate:
        players=players[1:]+players[:1]
        start_stack=start_stack[1:]+start_stack[:1]
    state = game(start_stack, len(players))
    return state, players

def create_new_phh(old_filename,new_filname):
    #準備新的牌局，從舊牌局繼承
    #讀取舊牌局
    game, state, players, others = read_hh(old_filename, prepare_final_stack=True)
    #開新局，移動按鈕
    n_state, n_players=next_state(game,players,state.stacks,rotate=True)
    #建立空的新局
    write_hh(new_filname, game, n_state, n_players, others, fix = False)

def auto_write_new_phh(filename,record_actions=[],smart=False):
    #開始記錄新局
    #讀取新局
    game, state, players, others = read_hh(filename, prepare_final_stack=False)
    #
    action_now=0
    #輸入步驟
    while state.status:
        if state.can_post_ante():
            state.post_ante()
        elif state.can_collect_bets():
            state.collect_bets()
        elif state.can_post_blind_or_straddle():
            state.post_blind_or_straddle()
        elif state.can_burn_card():
            state.burn_card('??')
        elif state.can_kill_hand():
            state.kill_hand()
        elif state.can_push_chips():
            state.push_chips()
        elif state.can_pull_chips():
            state.pull_chips()
        elif state.can_show_or_muck_hole_cards():
            state.show_or_muck_hole_cards()
        else:
            if len(record_actions)>action_now:
                action=record_actions[action_now]
                action_now+=1
            else:
                #print(state.can_check_or_call(),state.can_complete_bet_or_raise_to(),state.can_fold())
                action = input('Action: ')
                if action=="wtf":
                    break
            if smart:
                if action=="":
                    continue
                if state.can_check_or_call() or state.can_complete_bet_or_raise_to() or state.can_fold():
                    if action[0]=="d":
                        print("cant d",action)
                        continue
                if state.can_deal_board() or state.can_deal_hole():
                    if action[0]=="p":
                        print("cant p",action)
                        continue
                if action=="f" and state.can_fold():
                    state.fold()
                    continue
                if action=="cc" and state.can_check_or_call():
                    state.check_or_call()
                    continue
                if action[:3]=="cbr" and state.can_complete_bet_or_raise_to():
                    num=action[3:]
                    if action[-1:]=="m":
                        num=num[:-1]+"000000"
                    if action[-2:]==".5":
                        num=num[:-2]+"500000"
                    num=int(num)
                    if state.can_complete_bet_or_raise_to(num):
                        state.complete_bet_or_raise_to(num)
                        continue
                if (action[-1:]=="m" or action[-2:]==".5") and state.can_complete_bet_or_raise_to():
                    num=action
                    if action[-1:]=="m":
                        num=num[:-1]+"000000"
                    if action[-2:]==".5":
                        num=num[:-2]+"500000"
                    num=int(num)
                    if state.can_complete_bet_or_raise_to(num):
                        state.complete_bet_or_raise_to(num)
                        continue
                if len(action)==2:
                    if bool(re.match(r'[2-9ATJQK][hdsc]', action, re.IGNORECASE)):
                        if state.can_deal_board():
                            state.deal_board(action)
                            continue
                if len(action)==4:
                    if bool(re.match(r'[2-9ATJQK][hdsc][2-9ATJQK][hdsc]', action, re.IGNORECASE)):
                        if state.can_deal_hole():
                            state.deal_hole(action)
                            continue
                if len(action)==6:
                    if bool(re.match(r'[2-9ATJQK][hdsc][2-9ATJQK][hdsc][2-9ATJQK][hdsc]', action, re.IGNORECASE)):
                        if state.can_deal_board():
                            state.deal_board(action)
                            continue
            parse_action(state, action)
    #寫回並檢查
    write_hh(filename, game, state, players, others, fix = True)

def create_new_phh_from_eliminate(old_filename,new_filname,n_players=None,n_start_stack=None):
    #準備新的牌局，從舊牌局繼承
    #讀取舊牌局
    game, state, players, others = read_hh(old_filename, prepare_final_stack=True)
    #若不確定新位置，只確認
    if n_players is None:
        print(players,state.stacks)
        return
    #開新局，重新定義位置
    n_state, n_players=next_state(game,n_players,n_start_stack,rotate=False)
    #建立空的新局
    write_hh(new_filname, game, n_state, n_players, others, fix = False)

if __name__=="__main__":
    old_filename = "hand_record/11.phh"
    new_filename = "hand_record/12.phh"
    create_new_phh(old_filename,new_filename)
    # create_new_phh_from_eliminate(old_filename,new_filename)
    #編輯action
    record_actions = read_from_file("rec.txt")
    record_actions = parse_poker_log_all(record_actions)
    print(record_actions)
    auto_write_new_phh(new_filename,record_actions,smart=1)
'''
old_filename = "hand_record2/9.phh"
new_filename = "hand_record2/9_copy.phh"
d dh p1 Ac9s
d dh p2 Js3s
d dh p3 Kc7h
d dh p4 Qd2h
d dh p5 Qc6c
d dh p6 7s5d
d dh p7 Td6d
d dh p8 Ks2c
d dh p9 AsJh
p3 f
p4 f
p5 f
p6 f
p7 f
p8 f
p9 cbr 6000000
p1 cbr 24000000
p2 f
p9 cc
d db Kh3hQh
d db 2s
d db Jc
'''
'''from utg
Kc7hQd2hQc6c7s5dTd6dKs2cAsJh6mAc9s24mJs3s
cc
Kh3hQh
2s
Jc
'''