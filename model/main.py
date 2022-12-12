from model import MaceModel
import pandas as pd


def main():
    
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
    
    interventions = {}
    
    n = 2000
    stepsize = 50 * 12 
    model = MaceModel(n, seed=9, ethnicities=ethnicities, age_groups=age_groups,
        bmi25_dis_age_adults=bmi25_dis_age_adults, bmi25_dis_age_children=bmi25_dis_age_children, \
                bmi_dis_adults = bmi_dis_adults, bmi_dis_children=bmi_dis_children, income=income, \
                    smoking_age_adults=smoking_age_adults, smoked_adults = smoked_adults, smoking_children = smoking_children, interventions = interventions)
    
    #df_data_nodes= pd.DataFrame()
    #df_data_edges= pd.DataFrame()
    for i in range(stepsize):
        model.step()
    #     if (i % 12 == 0):
    #         data_nodes, data_edges = model.gephi_export()
    #         data_nodes.to_csv("D:/juju/lumc/model/gephi/nodes_" + str(i//12) + ".csv", index = False)
    #         data_edges.to_csv("D:/juju/lumc/model/gephi/edges_" + str(i//12) + ".csv", index = False)
    
    # data = model.get_datacollection()
    # data.to_csv("D:/juju/lumc/model/gephi/triangle.csv", index = False)

        
   #model.stats()
    
if __name__ == '__main__':
    main()
