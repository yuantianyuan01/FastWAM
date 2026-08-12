import random

def check_if_winning(a:list):
    '''Judge cards hepai. For excample:
        a=[1,2,3,4,4]
        man：1-9，sou：11-19，pin：21-29，east/west/south/north wind：31,33,35,37， red/green/white dragon：41,43,45。'''
    
    a=sorted(a)
    pais=list(range(1,10))+list(range(11,20))+list(range(21,30))+list(range(31,38,2))+list(range(41,46,2))
    for x in set(a):
        if a.count(x)>4:
            return False
        if x not in pais:
            print('Invalid parameter: The input hand type {} is out of range.\n Man: 1-9, Sou: 11-19, Pin: 21-29, East, South, West, North Winds: 31, 33, 35, 37, Red/Green/White Dragon: 41, 43, 45.'.format(x))
            return False

    if len(a)%3!=2:
        print('Invalid parameter: The number of cards is not correct.')
        return False
    
    double=[]
    for x in set(a):
        if a.count(x)>=2:
            double.append(x)
    #print(double)
    if len(double)==0:
        print('Invalid parameter: The number of pairs is not correct.')
        return False
    
    if len(a)==14:
        for x in set(a):
            if a.count(x) not in [2,4]:
                break
        else:
            return True

    if len(a)==14:
        gtws=[1, 9, 11, 19, 21, 29, 31, 33, 35, 37, 41, 43, 45]
        #print(gtws)
        for x in gtws:
            if 1<=a.count(x)<=2:
                pass
            else:
                break
        else:
            return True

    a1=a.copy()
    a2=[]
    for x in double:
        a1.remove(x)
        a1.remove(x)
        a2.append((x,x))
        for i in range(int(len(a1)/3)):
            if a1.count(a1[0])==3:
                a2.append((a1[0],)*3)
                a1=a1[3:]
            elif a1[0] in a1 and a1[0]+1 in a1 and a1[0]+2 in a1:
                a2.append((a1[0],a1[0]+1,a1[0]+2))
                a1.remove(a1[0]+2)
                a1.remove(a1[0]+1)
                a1.remove(a1[0])

            else:
                a1=a.copy()
                a2=[]
                break
        else:
            return True
    else:
        return False

def get_all_mahjongs():
    """
    get all mahjongs
    """
    man = [f"man_{i}" for i in range(1, 10)]
    sou = [f"sou_{i}" for i in range(1, 10)]
    pin = [f"pin_{i}" for i in range(1, 10)]
    dict2id = dict()
    for id, key in enumerate(man):
        dict2id[key] = id + 1
    for id, key in enumerate(sou):
        dict2id[key] = id + 11
    for id, key in enumerate(pin):
        dict2id[key] = id + 21 
    all_majhongs = man*4 + sou*4 + pin*4 
    random.shuffle(all_majhongs)
    return all_majhongs, dict2id
    
def generate_seven_pairs_mahjongs():
    """
    generate a list of mahjongs for a seven pairs hand
    """
    hands = []
    all_mahjongs, _ = get_all_mahjongs()
    for _ in range(6):
        pair = random.choice(all_mahjongs)
        hands.extend([pair, pair])
        for _ in range(2): all_mahjongs.remove(pair) # remove the pair from all_mahjongs
    winning_hand = random.choice(all_mahjongs)
    hands.append(winning_hand)
    return hands, [winning_hand]

def generate_nine_gates_mahjongs():
    """
    generate a list of mahjongs for a nine gates hand
    """
    suit = random.choice(['man', 'pin', 'sou'])
    hands = [f'{suit}_1'] * 3 + [f'{suit}_9'] * 3 + [f'{suit}_{i}' for i in range(2, 9)]
    winning_hands = [f'{suit}_{i}' for i in range(1, 10)]
    return hands, winning_hands

def generate_sequence_mahjongs(mahjongs):
    sequence = []
    start_mahjong = random.choice(mahjongs)
    while "8" in start_mahjong or "9" in start_mahjong:
        start_mahjong = random.choice(mahjongs)
    suite, value = start_mahjong.split('_')[0], int(start_mahjong.split('_')[1])
    if f"{suite}_{value+1}" in mahjongs and f"{suite}_{value+2}" in mahjongs:
        mahjongs.remove(start_mahjong)
        mahjongs.remove(f"{suite}_{value+1}")
        mahjongs.remove(f"{suite}_{value+2}") 
        sequence.extend([start_mahjong, f"{suite}_{value+1}", f"{suite}_{value+2}"])
    else:
        return generate_sequence_mahjongs(mahjongs)
    return sequence, mahjongs

def generate_triplet_mahjongs(mahjongs):
    triplet = []
    mahjong = random.choice(mahjongs)
    if mahjongs.count(mahjong) >= 3:
        for _ in range(3): mahjongs.remove(mahjong)
        triplet.extend([mahjong, mahjong, mahjong])
    else:
        return generate_triplet_mahjongs(mahjongs)
    return triplet, mahjongs

def generate_normal_hand_mahjongs():
    """
    generate a list of mahjongs for a normal hand
    """
    hands = []
    all_mahjongs, dict2id = get_all_mahjongs()
    pair = random.choice(all_mahjongs)
    hands.extend([pair, pair])
    for _ in range(2): 
        all_mahjongs.remove(pair)
    for _ in range(4):
        if random.random() < 0.5: # generate a sequence
            sequence, all_mahjongs = generate_sequence_mahjongs(all_mahjongs)
            hands.extend(sequence)
        else: #generate a triplet
            triplet, all_mahjongs = generate_triplet_mahjongs(all_mahjongs)
            hands.extend(triplet)
    
    drop = hands.pop(random.randint(0, len(hands)-1))
    winning_hands = set()
    # sequence_to_judge = [0] * 34
    # for hand in hands:
    #     sequence_to_judge[dict2id[hand]] += 1
    sequence_to_judge = [dict2id[hand] for hand in hands]
    all_candidate_majhongs = set(all_mahjongs.copy())
    for candidate_majhong in all_candidate_majhongs:
        seq = sequence_to_judge.copy()
        seq.append(dict2id[candidate_majhong])
        seq = sorted(seq)
        win = check_if_winning(seq)
        if win: 
            winning_hands.add(candidate_majhong)    
    return hands, list(winning_hands)

def generate_ready_hand_mahjongs():
    """
    generate a list of mahjongs for a ready hand
    """
    if random.random() < 0.1:
        hands, winning_hands = generate_seven_pairs_mahjongs()
    if random.random() < 0.2:
        hands, winning_hands = generate_nine_gates_mahjongs()
    else:
        hands, winning_hands = generate_normal_hand_mahjongs()
    return sorted(hands), sorted(winning_hands)