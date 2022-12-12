from typing import Counter
from model import MaceModel
import pandas as pd
import sys

def set_children_model(seef):
    
    ethnicities = {"Dutch": 43.8,
                   "Turkish": 7.6,
                   "Moroccan": 5.9,
                   "Hindustan": 3.7,
                   "Other": 39
                   }
    
    total= 549166
    
    age_groups = {(12, 18): 1}
    
    bmi25_dis_age_adults = { (18, 35): 0.312,
                          (35, 64): 0.532,
                          (64, 110): 0.577}

    smoking_age_adults = { (18, 35): 0.22,
                          (35, 64): 0.20,
                          (64, 110): 0.12}
    
    #people that used to smoke
    smoked_adults = 0.284
    #12-17: smoking, has smoked
    smoking_children = [0.064, 0.08]
    
    bmi25_dis_age_children = {(0, 2): 0.108,
                              (2, 5): 0.108,
                              (5, 9): 0.254,
                              (9, 14): 0.274,
                              (14, 18): 0.262}
        
    bmi_dis_children = {25: 0.14,
                        30: 0.093}
      
    bmi_dis_adults = {25: 0.535,
               30: 0.146}
    
    income = {"Turkish": {0: 0.761, 1: 0.21, 2: 0.029},
              "Hindustan": {0: 0.652, 1: 0.325, 2: 0.023},
              "Other": {0: 0.646, 1: 0.238, 2: 0.116},
              "Moroccan": {0: 0.867, 1: 0.12, 2: 0.013},
              "Dutch": {0: 0.548, 1: 0.386, 2: 0.066}        
    }
    
    n = 800
    
    interventions = {}

    model = MaceModel(n, seed=9876, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children, interventions = interventions)
    
    return model

def set_model(seed, agents, interventions = {}):
    
    ethnicities = {"Dutch": 43.8,
                   "Turkish": 7.6,
                   "Moroccan": 5.9,
                   "Hindustan": 3.7,
                   "Other": 39
                   }
    
    total= 549166
    
    age_groups = {(0, 4): 30248/total,
                  (5, 14): 61160/total,
                  (15, 19): 30972/total,
                  (20, 44): 202822/total,
                  (45, 64): 142551/total,
                  (65, 79): 61907/total,
                  (80, 109): 19506/total}
    
    bmi25_dis_age_adults = { (18, 35): 0.312,
                          (35, 64): 0.532,
                          (64, 110): 0.577}

    smoking_age_adults = { (18, 35): 0.22,
                          (35, 64): 0.20,
                          (64, 110): 0.12}
    
    #people that used to smoke
    smoked_adults = 0.284
    #12-17: smoking, has smoked
    smoking_children = [0.064, 0.08]
    
    #TODO TO FIX, because 2-5, some children have no bmi.
    bmi25_dis_age_children = {(0, 2): 0.108,
                              (2, 5): 0.108,
                              (5, 9): 0.254,
                              (9, 14): 0.274,
                              (14, 18): 0.262}
        
    bmi_dis_children = {25: 0.14,
                        30: 0.093}
      
    bmi_dis_adults = {25: 0.535,
               30: 0.146}
    
    income = {"Turkish": {0: 0.761, 1: 0.21, 2: 0.029},
              "Hindustan": {0: 0.652, 1: 0.325, 2: 0.023},
              "Other": {0: 0.646, 1: 0.238, 2: 0.116},
              "Moroccan": {0: 0.867, 1: 0.12, 2: 0.013},
              "Dutch": {0: 0.548, 1: 0.386, 2: 0.066}        
    }
    
    n = agents

    model = MaceModel(n, seed=seed, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children, interventions = interventions)
    
    return model

def check_birthdays(seed):
    model = set_model(seed, 1000)
    age_dict = {}
    for agent in model.schedule.agents:
        age_dict[agent.unique_id] = agent.age
        
    for _ in range(12):
        model.step()
        
    for agent in model.schedule.agents:
        if agent.unique_id not in age_dict:
            continue
        assert age_dict[agent.unique_id] == agent.age - 1, "age is incorrect"
    print("Birthday check success!")
    
def check_social_network(seed):
    model = set_model(seed, 1000)
    
    for _ in range(16):
        model.step()
        
    for agent in model.schedule.agents:
        for friend in agent.nodes:
            assert agent in friend.nodes, "friend does not consider agent friend - nodes"
            assert agent in friend.degrees.keys(), "degrees are off - nodes"
        for friend in agent.degrees.keys():
            assert agent in friend.nodes, "friend does not consider agent friend - degrees"
            assert agent in friend.degrees.keys(), "degrees are off - degrees"
            
def check_smoking_children(seed):
    model = set_children_model(seed)
    smoking = []
    former_smoking = []
    for agent in model.schedule.agents:
        if (agent.age >= 12) and (agent.age < 18):
            smoking.append(agent.smoking)
            former_smoking.append(agent.former_smoking)
    
    smoking = Counter(smoking)
    former_smoking = Counter(former_smoking)
    
    total = sum(count for count in smoking.values())
    print(total)
    print("smoking:", smoking[1]/total, "has smoked:", former_smoking[1]/total)
    
def check_medication_usage(seed):
    model = set_model(seed, 1000)
    
    for _ in range(12*5):
        model.step()
    
    medication_bp = 0
    medication_gluc = 0
    medication_chol = 0
    people = 0
    overweight = 0

    for agent in model.schedule.agents:
        if agent.age < 18 or model.agent_condition(agent):
            continue
        if (agent.bmi >= 25):
            overweight += 1
        medication_bp += agent.medication_bp
        medication_gluc += agent.medication_gluc
        medication_chol += agent.medication_chol
        people+=1
    
    print(" Overweight", overweight/people * 100, overweight)
    print("Medication blood pressure:", medication_bp/people * 100, medication_bp)
    print("Medication diabetes:", medication_gluc/people * 100, medication_gluc)
    print("Medication cholesterol:", medication_chol/people * 100, medication_chol) 
    
def check_smoking(seed):
    model = set_model(seed, 1000)
    
    for _ in range(20*12):
        model.step()
    
    smoking = 0
    former_smoking = 0
    people = 0
    for agent in model.schedule.agents:
        if agent.age < 18 or model.agent_condition(agent):
            continue
        if (agent.smoking == 1):
            smoking += 1
        if (agent.former_smoking == 1):
            former_smoking += 1
        people+=1
    
    print("Smoking", smoking/people * 100, smoking)
    print("Former smoking:", former_smoking/people * 100, former_smoking)
    
def check_weight(seed):
    model = set_model(seed, 1000)
    
    for _ in range(2):
        model.step()
    model.export_datacollection("testtest")
    
    model.stats()

def health_stats(seed):
    ethnicities = {"Dutch": 43.8,
                   "Turkish": 7.6,
                   "Moroccan": 5.9,
                   "Hindustan": 3.7,
                   "Other": 39
                   }
    
    age_groups = {(65, 79): 0.5,
                  (80, 109): 0.5}
    
    bmi25_dis_age_adults = { (18, 35): 0.312,
                          (35, 64): 0.532,
                          (64, 110): 0.577}

    smoking_age_adults = { (18, 35): 0.22,
                          (35, 64): 0.20,
                          (64, 110): 0.12}
    
    #people that used to smoke
    smoked_adults = 0.284
    #12-17: smoking, has smoked
    smoking_children = [0.064, 0.08]
    
    bmi25_dis_age_children = {(0, 2): 0.108,
                              (2, 5): 0.108,
                              (5, 9): 0.254,
                              (9, 14): 0.274,
                              (14, 18): 0.262}
        
    bmi_dis_children = {25: 0.14,
                        30: 0.093}
      
    bmi_dis_adults = {25: 0.835,
               30: 0.846}
    
    income = {"Turkish": {0: 0.761, 1: 0.21, 2: 0.029},
              "Hindustan": {0: 0.652, 1: 0.325, 2: 0.023},
              "Other": {0: 0.646, 1: 0.238, 2: 0.116},
              "Moroccan": {0: 0.867, 1: 0.12, 2: 0.013},
              "Dutch": {0: 0.548, 1: 0.386, 2: 0.066}        
    }
    
    n = 10
    interventions = {}

    model = MaceModel(n, seed=seed, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children, interventions = interventions)
    
    for agent in model.agents:
        print("\033[1m age: ", agent.age, "bmi:", agent.bmi, "\033[0m")
        agent.medication_gluc = 0
        agent.medication_chol = 0
        agent.medication_syst = 0
        print("blood glucose -", "chol value -", "syst blood pressure -", "MACE risk")
        print("\033[4m without medication \033[0m")
        print(agent.gluc_value(), " - ", agent.chol_value(), " - ", agent.syst_value(), " - ", agent.MACE_risk)
        agent.medication_gluc = 1
        agent.medication_chol = 1
        agent.medication_syst = 1
        print("\033[4m with medication \033[0m")
        print(agent.gluc_value(), " - ", agent.chol_value(), " - ", agent.syst_value(), " - ", agent.MACE_risk)

def check_customdict(seed):
    ethnicities = {"Dutch": 43.8,
                "Turkish": 7.6,
                "Moroccan": 5.9,
                "Hindustan": 3.7,
                "Other": 39
                }
    
    total= 549166
    
    age_groups = {(0, 4): 30248/total,
                (5, 14): 61160/total,
                (15, 19): 30972/total,
                (20, 44): 202822/total,
                (45, 64): 142551/total,
                (65, 79): 61907/total,
                (80, 109): 19506/total}
    
    bmi25_dis_age_adults = { (18, 35): 0.312,
                        (35, 64): 0.532,
                        (64, 110): 0.577}

    smoking_age_adults = { (18, 35): 0.22,
                        (35, 64): 0.20,
                        (64, 110): 0.12}
    
    #people that used to smoke
    smoked_adults = 0.284
    #12-17: smoking, has smoked
    smoking_children = [0.064, 0.08]
    
    #TODO TO FIX, because 2-5, some children have no bmi.
    bmi25_dis_age_children = {(0, 2): 0.108,
                            (2, 5): 0.108,
                            (5, 9): 0.254,
                            (9, 14): 0.274,
                            (14, 18): 0.262}
        
    bmi_dis_children = {25: 0.14,
                        30: 0.093}
    
    bmi_dis_adults = {25: 0.535,
            30: 0.146}
    
    income = {"Turkish": {0: 0.761, 1: 0.21, 2: 0.029},
            "Hindustan": {0: 0.652, 1: 0.325, 2: 0.023},
            "Other": {0: 0.646, 1: 0.238, 2: 0.116},
            "Moroccan": {0: 0.867, 1: 0.12, 2: 0.013},
            "Dutch": {0: 0.548, 1: 0.386, 2: 0.066}        
    }
    
    n = 1

    model = MaceModel(n, seed=seed, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children)
    
    for agent in model.schedule.agents:
        for i in range(20):
            agent.macerisks[i] = i
            
    for agent in model.schedule.agents:
        print(agent.macerisks)
        
def check_macedeaths(seed):
    model = set_model(seed, 1000)
    steps = 12 * 3
    
    for i in range(steps):
        model.step()
    
    print(model.deathkeeper)
    print(model.macedeathkeeper)
    
def check_targeted_intervention(seed):
    interventions = {"Targeted": [4, 50]}
    model = set_model(seed, 2000, interventions)
    steps = 2 * 12
    for i in range(steps):
        model.step()
        
    df = model.get_datacollection()
    print(df)
       
    #model.worst_community(n=1)

def test_model():
    seed = 4545
    check_targeted_intervention(seed)
#    check_macedeaths(seed)
#    check_customdict(seed)
#    health_stats(seed)
#    check_weight(seed)
#    check_smoking(seed)
#    check_medication_usage(seed)
#    check_social_network(seed)
#    check_smoking_children(seed)
#    check_birthdays(seed) 
    
if __name__ == '__main__':
    test_model()
