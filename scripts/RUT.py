# functions for RUT scoring
import numpy as np
import subprocess

def select_candidates(s, params0, k=5000, n=100, YC_cutoff=4, CG_cutoff = 1, filter_cutoff=3):
    # C/G ratio based filter
    if (s.count("C")+0.1)/(s.count("G")+0.1) < CG_cutoff:
        return []
    distances = []
    last_i = 0
    for i in range(len(s)-1):
        if s[i:i+2] in ["UC","CC"]:
            distances.append(i-last_i)
            last_i = i
    # YC dimer count based filter
    if len(distances) < YC_cutoff:
        return []
    # initial state
    last_states = [('^', 0, 10000)]
    last_lengths = [0]
    locations = np.cumsum(distances)
    for d in distances:
        states = [] 
        for (state, score, cumd), length in zip(last_states,last_lengths):
            if state[-1] in "01":
                if (state.count("1") >= 6) or (length>=45):
                    # already have 6 YC dimer 
                    #or distance between last and first selected YC dimer >= 45 -> current length ~ 45 + 10 = 55 -> if select one more YC dimer, the length ~ 65
                    candidate_states = ["R"]
                else:
                    if state[-1] == "0":
                        candidate_states = ["0","1"]
                    else:
                        candidate_states = ["0","1","R"]
            elif state[-1] in ["L","^"]:
                candidate_states = ["L","1"]
            else:
                candidate_states = ["R"]
            for st in candidate_states:                         
                current_state = state + st                        
                current_cumd = cumd + d      
                current_score = score                
                if st == "1":                
                    # only in state 1 the distance and score changed 
                    if current_cumd < len(params0["D2S"]):
                        # with in range of provided distance to score mapping
                        current_score += params0["D2S"][current_cumd]
                        current_cumd = 0
                    elif current_cumd >= 10000:
                        # first YC dimer, add reward for initialization
                        current_score += params0["IR"]
                        current_cumd = 0
                    else:
                        # a strong panelty for large YC-YC distance 
                        current_score += params0["GP"]
                        current_cumd = 0
                states.append((current_state, current_score, current_cumd))
        last_states = sorted(states,key=lambda x:-x[1])[:k]
        last_lengths = []
        for state, score, cumd in last_states:
            picked_locations = get_picked_locations(locations,state[1:])
            if len(picked_locations) > 0:
                length = picked_locations[-1] - picked_locations[0]
            else:
                length = 0
            last_lengths.append(length)
    if len(last_states) == 0:
        return []
    masks = []
    records = []
    for last_state in last_states:
        if len(records) == n:
            break
        final_score = last_state[1]
        if final_score < filter_cutoff:
            break
        picked_locations = get_picked_locations(locations,last_state[0][1:])
        if len(picked_locations) < YC_cutoff:
            continue
        mask = np.zeros(len(s),dtype=int)
        mask[picked_locations[0]:picked_locations[-1]] = 1
        skip = False
        for m in masks:
            # skip intervals that largely overlap with former ones
            if ((m == 1) & (mask == 1)).sum()/((m == 1) | (mask == 1)).sum() > 0.5:
                skip = True
        if not skip:
            masks.append(mask)
            records.append((round(final_score,4), last_state[0][1:], picked_locations))
    return records


def get_picked_locations(locations,state):
    picked_locations = []
    for location, state in zip(locations,state):
        if state == "1":
            picked_locations.append(location+2)
    return picked_locations

def energy(sequence):
    cmd = ["/apps/home/lulab_jinyunfan/qhsky1/miniconda/envs/bioinfo-env/bin/RNAfold"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    lines = proc.communicate(sequence.encode())[0].decode()
    line = lines.strip().split("\n")[-1]
    energy = line[line.rfind("(")+1:line.rfind(")")]
    proc.wait()
    try:
        energy = float(energy)
    except:
        print(energy)
        energy = 0
    return energy

def extract_features(sequence, CG=True, mfe=False, dinuc=True):
    assert CG or mfe or dinuc
    # CG_ratio > 1 lead to a reward
    features = {}
    if CG:
        features['C/G'] = np.log2((sequence.count("C")+0.1)/(sequence.count("G")+0.1))
    if mfe:
        e = energy(sequence)    
        # energy > -10 kcal lead to a reward, the score never > 2
        features["energy"] = (e + 10)/5
    if dinuc:
        for i in range(len(sequence)-1):
            dimer = sequence[i:i+2]
            if dimer not in features:
                features[dimer] = 0
            features[dimer] += 1/len(sequence)
        features["YC"] = features.get("UC",0) + features.get("CC",0)
    return features

def scoring(features,weights):
    score = 0
    for k in weights:
        score += features.get(k,0)*weights[k]
    return round(score,4) 
