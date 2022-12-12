from model import MaceModel
import pandas as pd
from ema_workbench import Model, RealParameter, ScalarOutcome, ema_logging, perform_experiments, Constant, TimeSeriesOutcome, save_results, IntegerParameter
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
import timeit

def mace_model(seed=18, scenario=0, interventions=0):
    # https://github.com/BROSE-Uninc/SSF2021/blob/main/5-sa_demo_wolf_sheep.ipynb
    ethnicities = {"Dutch": 43.8,
                   "Turkish": 7.6,
                   #"Surinamese": 8.3 ,
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
    
    #immediate impact:
    bmi25_dis_age_adults = { (18, 35): 0.3,
                          (35, 64): 0.3,
                          (64, 110): 0.3}

    smoking_age_adults = { (18, 35): 0.22,
                          (35, 64): 0.20,
                          (64, 110): 0.12}
    
    #has smoked: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/83021NED/table?dl=1BB38
    # also look at cbs if there's groups with both former smokign and is smoking
    # -> they're mutually exlcusive in ELAN
    
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
    

    # argument: targeted_people = 50
    # frequency = 4    
    # if targeted_people == 1:
    #     targeted_people = 50
    #     frequency = 4
    # elif targeted_people == 2:
    #     targeted_people = 200
    #     frequency = 4
    # elif targeted_people == 3:
    #     targeted_people = 200
    #     frequency = 1
    # elif targeted_people == 4:
    #     targeted_people = 400
    #     frequency = 4
    # else:
    #     targeted_people = 400
    #     frequency = 1
    

    interventions = {}
    #
    #interventions = {"School": [[13, (12*10)+1, (12*20)+1, (12*30)+1], 12, 0.8]}
    #alternatively: smoking threshold of non-smokers
    #making it easier to stop: smoking_external_push_factor
    #interventions = {"Smoking": [[1, 13, (12*10)+1, (12*20)+1, (12*30)+1], (-0.3, 0.1),  3 ]}
    #interventions = {"Targeted": [frequency, targeted_people]}

    # if interventions == 0:
    #     interventions = {"Smoking": [range(1, 970 ,12), (-0.3, 0.1),  3 ]}
    # elif interventions == 1:
    #     interventions = {"Smoking": [range(1, 970 ,12), (-0.1, 0.1),  3 ]}
    # elif interventions == 2:
    #     interventions = {"Smoking": [range(1, 970 ,12), (-0.1, 0.1),  1 ]} 
    # elif interventions == 3:
    #     interventions = {"Smoking": [range(1, 970 ,3 * 12), (-0.1, 0.1),  6 ]}  

    # if interventions == 0:
    #     interventions = {"School": [range(1, 970 ,12), 2, 0.4]}
    # elif interventions == 1:
    #     interventions = {"School": [range(1, 970 ,12), 6, 0.4]}
    # elif interventions == 2:
    #     interventions = {"School": [range(1, 970 ,12*3), 5, 0.9]} 
    # elif interventions == 3:
    #     interventions = {"School": [range(1, 970, 6), 1, 0.3]}

    if scenario == 0:
        scenario = "standard smoking"
    elif scenario == 1:
        scenario = "more susceptible smoking"
    else:
        scenario ="more influence calories"
        
    n = 2000
    stepsize = 50 * 12 
    model = MaceModel(n, seed=seed, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children, interventions = interventions, scenario=scenario)
    
    for i in range(stepsize):
        model.step()
        
    # model.export_datacollection("targetedzz")

    df = model.get_datacollection()


    # model.stats()  

    # outcomes = wolf_sheep.datacollector.get_model_vars_dataframe()
    
    # # Return model outcomes
    # # below to be changed!
    # return {'TIME' : list(range(steps + 1)),
    #         "Wolves" : outcomes["Wolves"].tolist(),
    #         "Sheep" : outcomes["Sheep"].tolist()}

   
    return { "Year" : df["Year"].tolist(),
             "Adults smoking" : df["Adults smoking"].tolist(),
             "Youngsters smoking" : df["Youngsters smoking"].tolist(),
             "MACE risk": df["Mace risk"].tolist(),
             "Overweight Moroccan": df["overweight Moroccan"].tolist(),
             "Overweight Turkish": df["overweight Turkish"].tolist(),
             "Overweight Hindustan": df["overweight Hindustan"].tolist(),
             "Overweight Dutch": df["overweight Dutch"].tolist(),
             "Overweight Other": df["overweight Other"].tolist(),
             "Overweight 18-24": df["overweight 18-24"].tolist(),
             "Overweight 25-64": df["overweight 25-64"].tolist(),
             "Overweight 65+": df["overweight 65+"].tolist(),
             "Overweight 12-17": df["overweight 12-17"].tolist(),
             "Diabetes adults": df["Diabetes adults"].tolist(),
             "Hypertension adults": df["Hypertension adults"].tolist(),
             "Dyslipidemia adults": df["Dyslipidemia adults"].tolist(),
             "Diabetes youngsters": df["Diabetes youngsters"].tolist(),
             "Hypertension youngsters": df["Hypertension youngsters"].tolist(),
             "Dyslipidemia youngsters": df["Dyslipidemia youngsters"].tolist(),
             "MACE events": df["MACE events"].tolist(),
             "avg age MACE": df["avg age MACE"].tolist(),
             "low MACE risk Dutch": df["low MACE risk Dutch"].tolist(),
             "low MACE risk Turkish": df["low MACE risk Turkish"].tolist(),
             "low MACE risk Moroccan": df["low MACE risk Moroccan"].tolist(),
             "low MACE risk Hindustan": df["low MACE risk Hindustan"].tolist(),
             "low MACE risk Other": df["low MACE risk Other"].tolist(),
             "mid MACE risk Dutch": df["mid MACE risk Dutch"].tolist(),
             "mid MACE risk Turkish": df["mid MACE risk Turkish"].tolist(),
             "mid MACE risk Moroccan": df["mid MACE risk Moroccan"].tolist(),
             "mid MACE risk Hindustan": df["mid MACE risk Hindustan"].tolist(),
             "mid MACE risk Other": df["mid MACE risk Other"].tolist(),
             "high MACE risk Dutch": df["high MACE risk Dutch"].tolist(),
             "high MACE risk Turkish": df["high MACE risk Turkish"].tolist(),
             "high MACE risk Moroccan": df["high MACE risk Moroccan"].tolist(),
             "high MACE risk Hindustan": df["high MACE risk Hindustan"].tolist(),
             "high MACE risk Other": df["high MACE risk Other"].tolist(),
            #  "low MACE risk ethnicity": df["low MACE risk ethnicity"].tolist()[0],
            #  "mid MACE risk ethnicity": df["mid MACE risk ethnicity"].tolist(),
            #  "high MACE risk ethnicity": df["high MACE risk ethnicity"].tolist(),
             "low MACE risk mean age": df[ "low MACE risk mean age"].tolist(),
             "mid MACE risk mean age": df["mid MACE risk mean age"].tolist(),
             "high MACE risk mean age": df["high MACE risk mean age"].tolist(),
             "Amount 65+": df["Amount 65+"].tolist(),
             "Amount 25-65": df["Amount 25-65"].tolist(),
             "Amount 18-24": df["Amount 18-24"].tolist(),
             "Amount youngsters": df["Amount youngsters"].tolist(),
             "Amount total": df["Amount total"].tolist(),
             "All conditions": df["All conditions"].tolist(),
             "Average MACE risk": df["average MACE risk"].tolist(),
            #  "MACEs 18-24": df["MACEs 18-24"].tolist(),
            #  "MACEs 25-64": df["MACEs 25-64"].tolist(),
            #  "MACEs 65+": df["MACEs 65+"].tolist(),
             "Average bmi": df["Average bmi"].tolist(),
             "ages maces 0": df["ages maces 0"].tolist(),
             "ages maces 1": df["ages maces 1"].tolist(),
             "ages maces 2": df["ages maces 2"].tolist(),
             "ages maces 3": df["ages maces 3"].tolist(),
             "ages maces 4": df["ages maces 4"].tolist(),
             "ages maces 5": df["ages maces 5"].tolist(),
             "ages maces 6": df["ages maces 6"].tolist(),
             "ages maces 7": df["ages maces 7"].tolist(),
             "ages maces 8": df["ages maces 8"].tolist()
             }
    
if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    
    model = Model("MACEmodel", function=mace_model)
    
    model.uncertainties = [
        IntegerParameter("seed", 0, 2**32 - 1),
        IntegerParameter("scenario", 0, 2),
        # IntegerParameter("targeted_people", 1, 5),
        # IntegerParameter("interventions", 0, 3)
    ]
    
    #     interventions = {}
    #interventions = {"School": [[13, (12*10)+1, (12*20)+1, (12*30)+1], 12, 0.8]}
    #alternatively: smoking threshold of non-smokers
    #making it easier to stop: smoking_external_push_factor
    #what years, effectiveness ,duration (Effectiveness is a value between 0 and 0.1 standard)
    #interventions = {"Smoking": [[1, 13, (12*10)+1, (12*20)+1, (12*30)+1], (-0.3, 0.1),  3 ]}
    #interventions = {"Targeted": [4, 50]}
    
    # model.levers = [RealParameter(str(i), 0, 0.1) for i in range(model.time_horizon)]
    
    

    model.outcomes = [TimeSeriesOutcome("Year"),
                      TimeSeriesOutcome("Adults smoking"),
                      TimeSeriesOutcome("Youngsters smoking"),
                      TimeSeriesOutcome("MACE risk"),
                      TimeSeriesOutcome("Overweight Moroccan"),
                      TimeSeriesOutcome("Overweight Turkish"),
                      TimeSeriesOutcome("Overweight Hindustan"),
                      TimeSeriesOutcome("Overweight Dutch"),
                      TimeSeriesOutcome("Overweight Other"),
                      TimeSeriesOutcome("Overweight 18-24"),
                      TimeSeriesOutcome("Overweight 25-64"),
                      TimeSeriesOutcome("Overweight 65+"),
                      TimeSeriesOutcome("Overweight 12-17"),
                      TimeSeriesOutcome("Diabetes adults"),
                      TimeSeriesOutcome("Hypertension adults"),
                      TimeSeriesOutcome("Dyslipidemia adults"),
                      TimeSeriesOutcome("Diabetes youngsters"),
                      TimeSeriesOutcome("Hypertension youngsters"),
                      TimeSeriesOutcome("Dyslipidemia youngsters"),
                      TimeSeriesOutcome("MACE events"),
                      TimeSeriesOutcome("avg age MACE"),
                      TimeSeriesOutcome("low MACE risk Dutch"),
                      TimeSeriesOutcome("low MACE risk Turkish"),
                      TimeSeriesOutcome("low MACE risk Moroccan"),
                      TimeSeriesOutcome("low MACE risk Hindustan"),
                      TimeSeriesOutcome("low MACE risk Other"),
                      TimeSeriesOutcome("mid MACE risk Dutch"),
                      TimeSeriesOutcome("mid MACE risk Turkish"),
                      TimeSeriesOutcome("mid MACE risk Moroccan"),
                      TimeSeriesOutcome("mid MACE risk Hindustan"),
                      TimeSeriesOutcome("mid MACE risk Other"),
                      TimeSeriesOutcome("high MACE risk Dutch"),
                      TimeSeriesOutcome("high MACE risk Turkish"),
                      TimeSeriesOutcome("high MACE risk Moroccan"),
                      TimeSeriesOutcome("high MACE risk Hindustan"),
                      TimeSeriesOutcome("high MACE risk Other"),
                    #    TimeSeriesOutcome("low MACE risk ethnicity"),
                    #   TimeSeriesOutcome("mid MACE risk ethnicity"),
                    #   TimeSeriesOutcome("high MACE risk ethnicity"),
                      TimeSeriesOutcome("low MACE risk mean age"),
                      TimeSeriesOutcome("mid MACE risk mean age"),
                      TimeSeriesOutcome("high MACE risk mean age"),
                      TimeSeriesOutcome("Amount 65+"),
                      TimeSeriesOutcome("Amount 25-65"),
                      TimeSeriesOutcome("Amount 18-24"),
                      TimeSeriesOutcome("Amount youngsters"),
                      TimeSeriesOutcome("Amount total"),
                      TimeSeriesOutcome("All conditions"),
                      TimeSeriesOutcome("Average MACE risk"),
                    #   TimeSeriesOutcome("MACEs 18-24"),
                    #   TimeSeriesOutcome("MACEs 25-64"),
                    #   TimeSeriesOutcome("MACEs 65+"),
                      TimeSeriesOutcome("Average bmi"),
                      TimeSeriesOutcome("ages maces 0"),
                      TimeSeriesOutcome("ages maces 1"),
                      TimeSeriesOutcome("ages maces 2"),
                      TimeSeriesOutcome("ages maces 3"),
                      TimeSeriesOutcome("ages maces 4"),
                      TimeSeriesOutcome("ages maces 5"),
                      TimeSeriesOutcome("ages maces 6"),
                      TimeSeriesOutcome("ages maces 7"),
                      TimeSeriesOutcome("ages maces 8")
                      ]    

    # results = perform_experiments(model, 1)
    
    start_time = timeit.default_timer()
    with MultiprocessingEvaluator(model, n_processes=22) as evaluator:
        results = perform_experiments(model, 100, evaluator=evaluator)
    end_time = timeit.default_timer() - start_time   
    save_results(results, "D:/juju/lumc/model/experiments/exp_immediate2_impact.tar.gz")
    
    print(results)
    # print(end_time)
    # 3 uur