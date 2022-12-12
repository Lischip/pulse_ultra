from inspect import trace
from mesa import Model
from mesa.time import RandomActivation
from agent import MaceAgent, fine_and_gray, breslow, Ethnicity_multiplier
import numpy as np
from enum import Enum
from collections import Counter, defaultdict
import scipy.stats as stats
import pandas as pd
from statistics import mean
  
import copy

import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain


def output_division(n, d):
    return n/d if d else np.nan


class Z_DataCollection:
    """Custom datatype that is used to collect data
    """
    def __init__(self):
        self.data = []

    def __setitem__(self, k, v):
        self.data.append(copy.deepcopy((k, v)))
        
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Z_DataCollection):
            return self.data == other.data
        return False
    
    def __iter__(self):
       return DicIter(self)
    
class DicIter:
    """Custom class to iter over the Z_Datacollection objects
    
    Raises
    ------
    StopIteration

    """
    def __init__(self, z_DataCollection):
        self._lect = z_DataCollection.data
        self._class_size = len(self._lect)
        self._current_index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_index < self._class_size:
            if self._current_index < len(self._lect):
                member = self._lect[self._current_index] 
            else:
                member = self._stud[self._current_index - len(self._lect)]
            self._current_index += 1
            return member
        raise StopIteration

class MaceModel(Model):
    """Model used to explore intevention entry points to reduce the burden
    of CVD among women in the Hague.
    
    Attributes
    ----------
    exercise : list
               list of exercise frequencies    
    """

    class Homogeneity(Enum):
        """Enum for different values of homogeneity of the ethnicity of the agent in her friend circle
        """
        Dutch = 0.88412
        Turkish = 0.54503
        Moroccan = 0.45198
        Hindustan = 0.14899
        Other = 0.21829

    class Dutch(Enum):
        """ num for percentage of Dutch people in friend circles
        """
        Turkish = 0.35
        Moroccan = 0.37
        Hindustan = 0.87
        Other = 0.75

    class Avg_height(Enum):
        """Enum for the average height of women. First value is average
        final height, second is standard deviation, third is minimum height, 
        fourth is maximum height 
        """
        Dutch = [170.7, 6.3, 150, 185]
        Turkish = [161, 6.4, 144, 178]
        Moroccan = [162.8, 6.5, 144, 179]
        Hindustan = [159.6, 5.9, 142, 174]
        Other = [161, 6.2, 144, 178]
            
    # age range, never <1 per month, < 1 per week, 1 or 2 per week, >2 per week
    exercise = [[(12, 17), 0.11, 0.078, 0.111, 0.307, 0.392],
                      [(18, 25), 0.129, 0.148, 0.16, 0.319, 0.243],
                      [(26, 49), 0.163, 0.1, 0.215, 0.288, 0.224],
                      [(41, 55), 0.233, 0.076, 0.197, 0.258, 0.235],
                      [(56, 70), 0.348, 0.057, 0.137, 0.199, 0.259],
                      [(71, 80), 0.397, 0.057, 0.128, 0.248, 0.17],
                      [(81, 110), 0.5, 0.4, 0.1, 0, 0]]
    
    def fill_population(self, bmi25_dis_age_adults, bmi25_dis_age_children, \
        age_groups, ethnicities_num, adult_overweight_dis, child_overweight_dis, \
            income, smoking_age_adults, smoked_adults, activation_months = None):
        """Populates the model with agents. Note that smoking children are initialised
        in the agent class.

        Args:
            bmi25_dis_age_adults (dict): Keys are a tuple of
            an age range. Including first term, excluding last term. Item is
            fraction of overweight. From age 18.
            bmi25_dis_age_children (dict): Keys are a tuple of an
            age range. Including first term, excluding last term. Item is
            fraction of overweight. Up to age 18 (excluding age 18)
            ethnicities_num (dict): Keys are an ethnicity (string). Item
            is number of agents.
            adult_overweight_dis (tuple): (fraction of adults overweight, 
            fraction of adults obese)
            child_overweight_dis (tuple): (fraction of children overweight,
            fraction of children obese)
            smoking_age_adults (dict): Keys are a tuple of an age range.
            Including first term, excluding last term. Item is fraction of
            age range smoking.
            smoked_adults (float): Total fraction of people that have stopped smoking
            activation_months (ndarray, optional): Ndarray of activation months
            [0, 11] for each agent. Defaults to None (meaning all agents are
            immediately active).
        """
        for k, v in ethnicities_num.items():
            
            adult_ethnicity_temp = {key: list() for key in bmi25_dis_age_adults.keys()}
            child_ethnicity_temp = {key: list() for key in bmi25_dis_age_children.keys()}
            # getting the ages           
            ages_distr = Counter(self.random.choices(range(0,  len(age_groups)), weights=age_groups.values(), k=v))

            ages = []
            for age_k, age_v in ages_distr.items():
                if age_k == len(ages_distr) - 1:
                    ages += np.random.triangular(list(age_groups)[age_k][0], list(age_groups)[age_k][0], list(age_groups)[age_k][1] + 1, size=age_v).astype(int).tolist()     
                else: 
                    ages += self.random.choices(range(list(age_groups)[age_k][0], list(age_groups)[age_k][1] + 1), k=age_v)

            heights = stats.truncnorm.rvs((self.Avg_height[k].value[2] - self.Avg_height[k].value[0]) / self.Avg_height[k].value[1], (self.Avg_height[k].value[3] - self.Avg_height[k].value[0]) / self.Avg_height[k].value[1], loc=self.Avg_height[k].value[0], scale=self.Avg_height[k].value[1], size=v)

            #income
            incomes = self.random.choices(list(income[k].keys()), list(income[k].values()), k = v)
            
            #Creating agents and adding them to the schedule.
            num_adults = 0
            num_children = 0
            mommys = list()
            youngerthan12 = list()
            for _ in range(v):
                age = ages.pop(0)
                my_height, heights = heights[-1], heights[:-1]
                my_income, incomes = incomes[-1], incomes[:-1]
                a = MaceAgent(self.id, self, k, age=age, height=my_height, income=my_income)
                if (activation_months is not None):
                    a.activation, activation_months = activation_months[0], activation_months[1:]
                self.schedule.add(a)
                self.agents.append(a)
                
                self.id += 1

                #pooling agents of ages per ethnicity
                if age < 12:
                    # all youngerthan12 are not given a BMI, in case the model cannot find a mother for them (edge case)
                    youngerthan12.append(a)    
                if age >= 18:
                    num_adults += 1
                    ethnicity_temp = adult_ethnicity_temp

                    if age < 45+12:
                        mommys.append(a)
                else:
                    num_children += 1
                    ethnicity_temp = child_ethnicity_temp
                for keys in ethnicity_temp.keys():
                    if (keys[0] <= age) and (age < keys[1]): 
                        ethnicity_temp[keys].append(a)  



            nomom = youngerthan12.copy()
            mommy = None
            potential_moms = mommys.copy()
            #giving children < 12 year old a mother
            if (len(potential_moms) > 0):
                for child in nomom:
                    while True:
                        # we ran out of moms!
                        if (len(potential_moms) == 0):
                            mommy = None
                            break
                        random_index = np.random.randint(0, len(potential_moms))
                        mommy = potential_moms[random_index]
                        potential_moms.remove(mommy)
                        # we found a mom!
                        if mommy.number_children < 3:
                            break
                    if (mommy == None):
                        break
                    child.mother = mommy
                    child.income = mommy.income
                    mommy.number_children += 1
                    nomom.remove(child)
            
            # first we process adults, then children.         
            ethnicity_temp = (adult_ethnicity_temp, child_ethnicity_temp)
            number_agents = (num_adults, num_children)
            overweight_dis = (adult_overweight_dis, child_overweight_dis)
           
            adult_ages_overweight = dict()

            for keyz, valuez in adult_ethnicity_temp.items():
                adult_ages_overweight[keyz] = int(len(valuez) *  bmi25_dis_age_adults[keyz])
            
            child_ages_overweight = dict()
            
            for keyz, valuez in child_ethnicity_temp.items():
                child_ages_overweight[keyz] = int(len(valuez) *  bmi25_dis_age_children[keyz])
                
            ages_overweight = (adult_ages_overweight, child_ages_overweight) 
            
            smoking = dict()  
            for keyz, valuez in adult_ethnicity_temp.items():
                #0: has smoked, 1: has not smoked, 2: is smoking
                smoking[keyz] = self.random.choices([0,1,2], weights=[(1-smoking_age_adults[keyz]) * smoked_adults, (1-smoking_age_adults[keyz]) * (1- smoked_adults),  smoking_age_adults[keyz]], k=len(valuez))

            for ethnicity_temp, ages_overweight, num_agents, overweight_dis in zip(ethnicity_temp, ages_overweight, number_agents, overweight_dis):
        
                overweight_number =  sum([value for value in ages_overweight.values()])
                # get number of people overweight and obesitas
                overweight_dis = Counter(self.random.choices(range(0,2), weights=overweight_dis, k=overweight_number))
                healthy_bmis = np.random.triangular(18, 25, 25, num_agents - overweight_number) 
                       
                # get their actual BMI value
                overweight_bmis = np.random.uniform(25, 30, overweight_dis[0])
                overweight_bmis = np.append(overweight_bmis, np.random.triangular(30, 30, 70, overweight_dis[1]))
              
                #giving agents overweight bmi
                for key, value in ages_overweight.items():
                    counter = 0
                    while counter < value:
                        if (overweight_bmis.size == 0) or (len(ethnicity_temp[key]) == 0):
                            break
 
                        agent = ethnicity_temp[key].pop()
                        random_index = np.random.randint(0, overweight_bmis.size)
                        agent.set_bmi(overweight_bmis[random_index])
                        overweight_bmis = np.delete(overweight_bmis, random_index)
                        # configure smoking
                        if ethnicity_temp == adult_ethnicity_temp:
                            smoking_stat = smoking[key].pop()
                            #0: has smoked, 1: has not smoked, 2: is smoking
                            if (smoking_stat == 0):
                                agent.former_smoking = 1
                            elif(smoking_stat == 2):
                                agent.starts_smoking()
                        counter += 1

                # giving remaining agents healthy bmi
                for ethnicity in ethnicity_temp.keys():
                    if len(ethnicity_temp[ethnicity]) == 0:
                        continue
                    else:
                        while True:
                            if len(ethnicity_temp[ethnicity]) == 0:
                                break
                            agent = ethnicity_temp[ethnicity].pop()
                            random_index = np.random.randint(0, healthy_bmis.size)
                            agent.set_bmi(healthy_bmis[random_index])
                            healthy_bmis = np.delete(healthy_bmis, random_index)
                            if ethnicity_temp == adult_ethnicity_temp:
                                smoking_stat = smoking[ethnicity].pop()
                                #0: has smoked, 1: has not smoked, 2: is smoking
                                if (smoking_stat == 0):
                                    agent.former_smoking = 1
                                elif(smoking_stat == 2):
                                    agent.starts_smoking()


            # Give mothers a number of children (not in the model)
            children_counter = num_children
            mommy = None
            while children_counter != 0:           
                while True:
                    # stop condition
                    if (len(mommys) == 0):
                        break
                    random_index = np.random.randint(0, len(mommys))
                    mommy = mommys[random_index]
                    mommys.remove(mommy)
                    if mommy.number_children < 3:
                        mommy = None
                        break
                    else:
                        mommy = None
                # stop condition: ran out of mothers
                if (mommy == None):
                    break                    

                birthed_number = np.random.randint(1, 4-mommy.number_children)
                
                if children_counter - birthed_number < 0:
                    birthed_number = children_counter             
                mommy.number_children += birthed_number
                children_counter -= birthed_number
                
    def smoking_trend(self, x):
        """Generate the trend of smoking over the years

        Args:
            x (ndarray): array of steps, so array([0,1,2,3,4]) if the model 
            will run only 5 steps

        Returns:
            float: Likelihood of being a smoker
        """
        eq = (1/2) * np.sin((1/40) * x-1.57) + np.random.normal(scale=0.1, size=len(x)) + 0.5
        return np.where(eq < 0, 0, eq)
            
    def __init__(self, n, bmi25_dis_age_adults, bmi25_dis_age_children, \
        bmi_dis_adults, bmi_dis_children, income, smoking_age_adults, smoked_adults, \
            smoking_children, seed = 12345, ethnicities = {"Dutch": 100}, \
                age_groups = {(0, 110): 1}, interventions = {}, \
                    scenario="standard smoking", *args, **kwargs):
        """Initialisation

        Args:
            n (int): number of agents the model starts running with
            bmi25_dis_age_adults (dict): Keys are a tuple of
            an age range. Including first term, excluding last term. Item is
            fraction of overweight. From age 18.
            bmi25_dis_age_children (dict): Keys are a tuple of an
            age range. Including first term, excluding last term. Item is
            fraction of overweight. Up to age 18 (excluding age 18)
            bmi_dis_children (dict): Keys are "overweight" and "obese".
            Items are fractions that are overweight and obese respectively
            income (dict): Keys are ethnicities. Items are dicts of income 
            distributions of income categories 0, 1 and 2, eg.
            "Dutch": {0: 0.333, 1: 0.333, 2: 0.333}
            smoking_age_adults (dict): Keys are a tuple of an age range.
            smoked_adults (float): Total fraction of people that have stopped smoking
            seed (int, optional): Seed of the model. Defaults to 12345.
            ethnicities (dict, optional): Dictionary with the distributions per
            ethnicity . Defaults to {"Dutch": 100}.
            interventions (dict, optional): Keys are the name of the intervention.
            Items are associated values. Defaults to {}.
            scenario (str, optional): scenario
        """
        super().__init__(seed=seed)
        # Why is the mesa package so retarded?
        super().reset_randomizer(seed)
        self.agents = []
        self.schedule = RandomActivation(self)
        self.n = n
        
        #all the seeds, haha. Orchard time.
        #below is used in agent
        self.seed = seed
        np.random.seed(seed=seed)
        self.rng =  np.random.default_rng(seed)       
    
        self.smoking_age_adults = smoking_age_adults
        self.smoked_adults = smoked_adults
        self.smoking_children = smoking_children + [1 - sum(smoking_children)] 

        #sort dictionary
        eth_distr = self.random.choices(range(0,  len(ethnicities)), weights=ethnicities.values(), k=n)
        ethnicities_num = []
        
        self.dataCollector = Z_DataCollection()
        self.deathkeeper = dict()
        self.macedeathkeeper = dict()
        self.mace_events = []
        #get the nationalities of all our agents
        for i in eth_distr:
            ethnicities_num += [list(ethnicities)[i]]
            
        self.deathlist = set()
        
        self.bmi_dis_adults = bmi_dis_adults
        self.bmi_dis_children = bmi_dis_children
        
        #calculate distribution of overweight vs obesitas
        total = sum(self.bmi_dis_adults.values())
        adult_overweight_dis = [x/total for x in self.bmi_dis_adults.values()]

        total = sum(self.bmi_dis_children.values())
        child_overweight_dis = [x/total for x in self.bmi_dis_children.values()]   
        
        self.bmi25_dis_age_adults = bmi25_dis_age_adults
        self.bmi25_dis_age_children = bmi25_dis_age_children
        
        self.max_calories = 500
        self.smoking_quitting_range = (0.6, 0.8)
        self.smoking_starting_rage = (0.063, 0.7) 
        self.smoking_trend = self.smoking_trend(np.arange(0,80*12))
        self.susceptibility = (0, 1)
        if scenario == "standard smoking":
            self.smoking_quitting_range = (0.6, 0.8)
        elif scenario == "more susceptible smoking":
            self.smoking_quitting_range = (0.4, 0.6)
            self.smoking_starting_rage = (0.063, 0.4) 
        elif scenario == "more influence calories":
            self.max_calories = 1000
            self.susceptibility = (0, 1.2)
        
        self.id = 0
        
        self.income = income
        
        self.fill_population(self.bmi25_dis_age_adults, self.bmi25_dis_age_children, age_groups, Counter(ethnicities_num), adult_overweight_dis, child_overweight_dis, income, smoking_age_adults, smoked_adults)
        
        self.interventions = interventions
        
        # matchmaking
        self.matchmake_friends()
        
    def exercise_amount(self, age):
        """Function to return probabilities of exercise intensity based on age

        Args:
            age (int): age (in years)

        Returns:
            list: Probabilities of exercise intensity
        """
        for values in self.exercise:
            if (age >= values[0][0]) and (age <= values[0][1]):
                return values[1:]

    def total_fertility_rate(self, year):
        """Returns the fertility rate per year

        Args:
            year (int): year

        Returns:
            float: fertitity rate of that year
        """
        if year >= 10:
            return (45.2/2/1000) * np.exp(0.019319122903085854*10)
        else:
            return (45.2/2/1000) * np.exp(0.019319122903085854*year)

      
    @property 
    def looking_for_friends(self):
        """Method that returns the list of agents that are still looking for
        friends.

        Returns:
            list: list of agents that are looking for friends
        """
        socialising = []

        for a in self.schedule.agents:
            if (a.age < 12) or self.agent_condition(a) or (a.unintended_wl):
                continue            
            if (a.weight == None):
                exit()
            friends_needed = dict( [(0, a.friends_eth[0] - a.current_friends[0]),  (1, a.friends_eth[1] - a.current_friends[1]), (2, a.friends_eth[2] - a.current_friends[2])])

            if (friends_needed[0] > 0) or \
                (friends_needed[1] > 0) or \
                    (friends_needed[2] > 0):
                        socialising.append(a)

        return socialising
    
    def befriending(self, agent, candidate, friends_needed):
        """Checks if two agents can befriend each other

        Args:
            agent (_type_): friend one
            candidate (_type_): friend two
            friends_needed (dict): the friends needed by agent

        Returns:
            dictionary, 0/1: (updated) friends_needed dictionary and a binary to
            indicate failure or success respectively
        """
        degree = self.random.triangular(0.1, 1, 1)
 
        if (agent.unintended_wl) or (candidate.unintended_wl) or \
            (agent == candidate) or abs(candidate.age - agent.age) > 10 or \
                (candidate in agent.nodes) or candidate.dead or \
                    not (agent.tolerate(candidate)) or \
                        not (candidate.tolerate(agent)):
            return friends_needed, 0
  
        if (abs(agent.smoking_desire -candidate.smoking_desire) > 0.2):
            return friends_needed, 0        
                
        if candidate.ethnicity == agent.ethnicity and friends_needed[0] > 0 and \
            (candidate.current_friends[0] < candidate.friends_eth[0]):
            friends_needed[0] -= 1
        elif candidate.ethnicity == "Dutch" and friends_needed[1] > 0 and \
            (candidate.current_friends[2] < candidate.friends_eth[2]):
            friends_needed[1] -= 1
        elif candidate.ethnicity not in ("Dutch", agent.ethnicity) and friends_needed[2] > 0:
            if (agent.ethnicity == "Dutch") and (candidate.current_friends[1] >= candidate.friends_eth[1]):
                return friends_needed, 0            
            elif (agent.ethnicity != candidate.ethnicity) and (candidate.current_friends[2] >= candidate.friends_eth[2]):
                return friends_needed, 0
            friends_needed[2] -= 1
        else:
            return friends_needed, 0
        
        agent.nodes.append(candidate)
        candidate.nodes.append(agent)
        agent.degrees[candidate] = degree
        candidate.degrees[agent] = degree

        return friends_needed, 1

        
     
    def matchmake_friends(self):
        """Method for all agents on the schedule to find friends
        """
        #update self_socialising
        socialising = self.looking_for_friends
                    
        while(socialising):
            # makeshift pop
            a, socialising = socialising[0], socialising[1:]
            

            friends_needed = dict( [(0, a.friends_eth[0] - a.current_friends[0]),  (1, a.friends_eth[1] - a.current_friends[1]), (2, a.friends_eth[2] - a.current_friends[2])])
            max_small_world_friends = int(sum(friends_needed.values()) * self.random.uniform(0, 1))
            
            # Theoretically, women in the end of the list may already have all the friends they need
            # due to women nearer to the start befriending them.
            if (friends_needed[0] == 0) and (friends_needed[1] == 0) and (friends_needed[2] == 0):
                continue
            
            # first check if I can befriend friends of my friends

            friends = a.nodes.copy()

            #small world effect
            if (max_small_world_friends > 0):
                for _ in range(len(friends)):
                    if max_small_world_friends == 0:
                        break
                    friend = self.random.choice(friends)
                    friends.remove(friend)
                    via_friends = friend.nodes.copy()
                    for _ in range(len(via_friends)):
                        if max_small_world_friends == 0:
                            break
                        via_friend = self.random.choice(via_friends)
                        via_friends.remove(via_friend)
                        friends_needed, result =  self.befriending(a, via_friend, friends_needed)
                    max_small_world_friends -= result

            candidates = socialising.copy()
            while (candidates):
                # select a random candidate!
                candidate = self.random.choice(candidates)
                candidates.remove(candidate)
          

                friends_needed, _ = self.befriending(a, candidate, friends_needed)

                via_friends = candidate.nodes.copy()
                             
                #small world effect
                if (max_small_world_friends > 0):
                    for _ in range(len(friends)):
                        if max_small_world_friends == 0:
                            break
                        friend = self.random.choice(friends)
                        friends.remove(friend)
                        via_friends = friend.nodes.copy()
                        for _ in range(len(via_friends)):
                            if max_small_world_friends == 0:
                                break
                            via_friend = self.random.choice(via_friends)
                            via_friends.remove(via_friend)
                            friends_needed, result =  self.befriending(a, via_friend, friends_needed)
                        max_small_world_friends -= result
                    
                if friends_needed[0] == 0 and friends_needed[1] == 0 and friends_needed[2] == 0:
                    continue
                            

    
    def remove_surplus_friends(self):
        """Remove surplus of friends for all agents on the schedule
        """
        for agent in self.schedule.agents:
            if (agent.age < 12) or agent.dead or self.agent_condition(agent):
                continue

            for dead in self.deathlist:
                if dead in agent.nodes:
                    agent.nodes.remove(dead)
                    agent.degrees.pop(dead, "Error")
                    dead.nodes.remove(agent)
                    dead.degrees.pop(agent, "Error")
            
            
            #remove elderly
            for friend in agent.nodes:
                if (friend.unintended_wl):
                    friend.degrees.pop(agent)
                    agent.degrees.pop(friend)
                    agent.nodes.remove(friend)
                    friend.nodes.remove(agent)

            agent_current_friends = agent.current_friends
                
            if (agent_current_friends[0] > agent.friends_eth[0]) or \
            (agent_current_friends[1] > agent.friends_eth[1]) or \
            (agent_current_friends[2] > agent.friends_eth[2]):
                               
                # now update nodes
                temp_counter = dict([(0, 0), (1,0), (2, 0)])
                to_delete= list()
                for friend in agent.nodes:
                    if (friend.ethnicity == agent.ethnicity):
                        temp_counter[0] += 1
                        if temp_counter[0] > agent.friends_eth[0]:
                            to_delete.append(friend)
                    elif (friend.ethnicity == "Dutch"):
                        temp_counter[1] += 1
                        if temp_counter[1] > agent.friends_eth[1]:
                            to_delete.append(friend)   
                    else:
                        temp_counter[2] += 1
                        if temp_counter[2] > agent.friends_eth[2]:
                            to_delete.append(friend)

                for friend in to_delete:
                    friend.degrees.pop(agent)
                    agent.degrees.pop(friend)
                    agent.nodes.remove(friend)
                    friend.nodes.remove(agent)


    def population_growth(self):
        """Method that adds agents (immigrants) to match the projected population growth
        """
        projection = int((-0.15917403 * np.exp(-0.0053049*self.schedule.steps) + 1.18092669) * self.n)
        agents_needed =  projection - len(self.schedule.agents)
        
        if agents_needed < 1:
            return
        
        time = min(self.schedule.steps, 25*12)
        dutch = int((-0.0006304 * time + 0.4388) * agents_needed)
        turkish = int((1.484e-05 * time + 0.07605) * agents_needed)
        moroccan = int((3.059e-05 * time + 0.05952) * agents_needed)
        hindustani = int((-3.535e-05 * time + 0.03688) * agents_needed)
        other = agents_needed - dutch - turkish - moroccan - hindustani

        ethnicities = {"Dutch": dutch,
                       "Turkish": turkish,
                       "Moroccan": moroccan,
                       "Hindustan": hindustani,
                       "Other": other
                       }
        
        ages = np.random.randint(low=0, high=65, size = agents_needed)
        
        age_groups = {(0,4): 0,
                      (5, 14): 0,
                      (15, 19): 0,
                      (20,44): 0,
                      (45, 65): 0}
        for age in ages:
            for key in age_groups.keys():
                if (age >= key[0]) and (age <= key[1]):
                    age_groups[key] += 1
        
        for key in age_groups.keys():
            age_groups[key] /= len(ages)

        
        activation_months = np.random.randint(0, 11, size = agents_needed)
        activation_months += self.schedule.steps
        
        #calculate distribution of overweight vs obesitas
        total = sum(self.bmi_dis_adults.values())
        adult_overweight_dis = [x/total for x in self.bmi_dis_adults.values()]
        total = sum(self.bmi_dis_children.values())
        child_overweight_dis = [x/total for x in self.bmi_dis_children.values()]  
        
        self.fill_population(self.bmi25_dis_age_adults, self.bmi25_dis_age_children, age_groups, ethnicities, adult_overweight_dis, child_overweight_dis, income = self.income, smoking_age_adults= self.smoking_age_adults, smoked_adults = self.smoked_adults, activation_months = activation_months)
            
                
    def influence_network(self):
        """Method that lets agents influence each other
        """
        for agent in self.schedule.agents:
            num_friends = len(agent.nodes)
            if self.agent_condition(agent) or (num_friends == 0) or (agent.age < 12) or agent.dead:
                continue
            
            exercise = 0
            intake = 0
            smoking_desire = 0
            quitting_desire = 0
            denom = 0
            
            for friend, degree in agent.degrees.items():
                intake += (degree * friend.caloric_intake)
                exercise += (degree * friend.expenditure)
                smoking_desire += (degree * friend.smoking)
                quitting_desire += (degree * friend.former_smoking)
                denom += degree
            
            agent.expenditure = exercise/denom
            
            agent.smoking_desire = smoking_desire/denom - agent.external_smoking_push_factor
            
            # max 500 calories deficit/increase

            difference = max(-self.max_calories, min(self.max_calories, (intake/denom) - agent.caloric_intake))
            agent.caloric_change(difference)


    @property
    def number_of_vertile(self):
        """Method that returns agents that could potentially become mothers

        Returns:
            list: Agents that could potentially become mothers
        """
        vertile = []
        for agent in self.schedule.agents:
            if (agent.age <12) or (agent.dead):
                continue
            
            if (agent.age >= 20) and (agent.age < 50) and (agent.number_children < 3) and not agent.pregnant:
                vertile.append(agent)
        return vertile.copy()            
 
    def determine_pregnancies(self, year):
        """Returns the number of agent that need to get pregnant for the year
        passed in the argument. Function is based on projection

        Args:
            year (int): the current year (starting from 0)
        """
        
        vertile_women = self.number_of_vertile
        birth_months = []
        num_births = int(self.total_fertility_rate(year) * len(vertile_women))
        
        for _ in range(num_births):
            if (year == 0): # here we determine for the first year IN the first year
                birth_months.append(self.random.randint(0, 11))
            else: # so here we determine the births in year n+1, which means people may already be pregnant in year n
                birth_months.append(self.random.randint(12, 23))
        
        while True:
            if (len(birth_months) == 0):
                return
            
            random_index = np.random.randint(0, len(vertile_women))
            pregnant = vertile_women[random_index]
            vertile_women.remove(pregnant)
 
            pregnant.pregnant = True
            pregnant.pregnancy_countdown = birth_months.pop()
              
    def gephi_export(self):
        """Method that allows a step to be exported to gephi

        Returns:
            dataframe, dataframe : Dataframe with nodes, dataframe with edges
        """
        data_nodes = {"id": [], "age": [], "ethnicity": [], "bmi": []}
        data_edges = {"source": [], "target": [], "weight": []}
        for agent in self.schedule.agents:
            if agent.age < 12 or self.agent_condition(agent):
                continue
            data_nodes["id"].append(agent.unique_id)
            data_nodes["age"].append(agent.age)
            data_nodes["ethnicity"].append(agent.ethnicity)
            data_nodes["bmi"].append(agent.bmi)

            data_edges["source"].append(agent.unique_id)
            data_edges["target"].append([key.unique_id for key in agent.degrees.keys()])
            data_edges["weight"].append([value for value in agent.degrees.values()])
            
        data_nodes = pd.DataFrame.from_dict(data_nodes)
        data_edges = pd.DataFrame.from_dict(data_edges)
        data_edges = data_edges.explode(["target", "weight"], ignore_index=True)
        return data_nodes, data_edges
          
    def step(self):
        """Advance model one step
        """
        # Every year, even year 0
        if (self.schedule.steps % 12 == 0) and (self.schedule.steps//12 != 80):
            self.determine_pregnancies((self.schedule.steps//12) + 1)
            
        if "School" in self.interventions:
            if self.schedule.steps in self.interventions["School"][0]:
                self.school_intervention(self.interventions["School"][1],self.interventions["School"][2])
        if "Smoking" in self.interventions:
            if self.schedule.steps in self.interventions["Smoking"][0]:

                self.smoking_intervention(months= self.interventions["Smoking"][2], value= self.interventions["Smoking"][1])
        if ("Targeted" in self.interventions) and ((self.schedule.steps == 12) or (((((self.schedule.steps - 12) % 12) * self.interventions["Targeted"][0]) == 0) and (self.schedule.steps == 12))) :
            self.targeted_intervention(self.interventions["Targeted"][1])

        self.influence_network()
        # update friends
        self.remove_surplus_friends()
        # add new friends.
        self.matchmake_friends()

        self.schedule.step() # all agents move now
        
        # Every year, even year 0
        if (self.schedule.steps % 12 == 0) and (self.schedule.steps//12 != 80):
            self.population_growth()
                
        
        #remove deathlist from schedule
        for death in self.deathlist:
            self.schedule.remove(death)
    
        self.deathlist = set()
        if (self.schedule.steps % 12 == 0):
            self.datacollection()
            self.macedeaths = []
            self.mace_events = []
        
    def agent_condition(self, agent):
        """Method that determines whether an agent is active or not

        Args:
            agent (MaceAgent): agent

        Returns:
            Bool: Returns true if agent is active, false if inactive
        """
        return (agent.dead) or (agent.activation != -1) or (agent.bmi == None)
    
    
    def school_intervention(self, months, fraction):
        """Method that processes the school intervention

        Args:
            months (int): The amount of months the intervention is active
            fraction (float): The fraction of children that will be reached
        """
        for agent in self.schedule.agents:
            if (self.agent_condition(agent)) or (agent.age > 16) or (agent.age < 12):
                continue
            else:
                if (self.random.uniform(0, 1) < fraction):
                    agent.school_intervention(0.9, 0.5, months)
                    
    def smoking_intervention(self, months, value):
        """Method that processes the smoking intervention

        Args:
            months (int): The number of months it is active
            value (array): New lower and upper limit of starting to smoke
        """
        for agent in self.schedule.agents:
            if (self.agent_condition(agent)):
                continue
            agent.change_external_smoking_push_factor(value, months)
            
    def worst_community(self, minimum = 1):
        """Method that finds the communities that are worst off and returns these.
        Part of the targeted intervention.

        Args:
            minimum (int, optional): Minimum number of communities that need to be returned
            
        Returns:
            array of arrays: clusters with agents that are worst off.
        """
        def MACE_risk(age, income, ethnicity):
            """Method to determine a somewhat optimal risk function for an agent

            Args:
                age (int): Age of an agent
                income (int): 0, 1 or 2 to match the income category
                ethnicity (str): ethnicity

            Returns:
                float: Optimal 10-year first ever MACE risk
            """
            gluc = 3.9
            chol = 8
            syst = 120
            x = np.array([age, syst, gluc, chol, \
                0, 0, 0, \
                    0, 0, (1 if income == 1 else 0), \
                        (1 if income == 2 else 0), age * syst, \
                            age * gluc, age * chol, \
                                age * 0, age * 0, \
                                    age * 0, age * 0, \
                                        age * 0])
            
            
            x = np.multiply(x, fine_and_gray)
            x = sum(x)
            return ((1-np.exp(-np.cumsum(breslow * np.exp(x)))[119]) * Ethnicity_multiplier[ethnicity].value)
            
        agent_network = {}

        to_remove = []

        for agent in self.schedule.agents:
            if self.agent_condition(agent):
                continue
            risk = agent.macerisks[self.schedule.steps//12 + 10]
            if risk == -1:
                to_remove.append(agent)
                continue
            agent_network[agent] = {k:v for k,v in agent.degrees.items()}

        df = pd.DataFrame(pd.Series(agent_network.keys()).rename("source"))

        df = pd.concat([df, pd.Series([list(v.keys()) for v in agent_network.values()]).rename("target")], axis=1)
        df = pd.concat([df, pd.Series([list(v.values()) for v in agent_network.values()]).rename("weight")], axis=1)

        df =  df.explode(["target", "weight"])

        
        df = df.reset_index()
        df = df[df['target'].notna()]
       
        df = df[~df.target.isin(to_remove)]
        df = df[~df.source.isin(to_remove)]
   
        G=nx.from_pandas_edgelist(df, "source", "target", ["weight"])
       
        partition = community_louvain.best_partition(G, resolution = 0.65)
      
        agents = defaultdict(list)
        for k, v in partition.items():
                agents[v].append(k)
   
        risks = []

        for k,v in agents.items():
            risks.append(mean([ max(0, agent.macerisks[self.schedule.steps//12 + 10] - MACE_risk(agent.age, agent.income, agent.ethnicity)) for agent in v]))
       
        worst_clusters = sorted(zip(risks, agents), reverse=True)
        
        counter = 0
        clusters = []
        for cluster in worst_clusters:
            while counter < minimum:
                clusters.append( agents[cluster[1]])
                counter += len(agents[cluster[1]])

        return clusters   
    
    def targeted_intervention(self, number_of_people = 50):
        """Method to function tohe targeted intervention

        Args:
            number_of_people (int, optional): The number of people that will be
            affected by the intervention. Defaults to 50.
        """
        communities = self.worst_community(number_of_people)
       
        smoking = 0
        overweight = 0
        counter = 0   
        for community in communities:         
            for agent in community:
                if (agent.smoking == 1):
                    smoking += 1
                if (agent.bmi >= 25):
                    overweight += 1
                    
            if smoking >= overweight:
                def intervention(agent):
                    agent.smoking_desire -= self.random.uniform(0.1, 1.4) 
            else:
                def intervention(agent):
                    agent.caloric_intake = max(agent.metabolic_rate, agent.caloric_intake - self.random.uniform(300, 500))
                    agent.expenditure =  self.random.uniform(1.375, 1.55) if (agent.expenditure < 1.375) else agent.expenditure
                                                        
            while (counter < number_of_people) and (len(community) > 0):
                random_index = np.random.randint(0, len(community))
                target = community[random_index]
                community.remove(target)
                intervention(target)
            
            
            if counter == number_of_people:
                break
            
        
    
    def datacollection(self):
        """Method that collects data
        """
        smoking = 0
        former_adult_smoking = 0
        former_young_smoking = 0
        young_smoking = 0
        young_people = 0
        adults = 0
        diabetes = 0
        dyslipidemia = 0
        hypertension = 0
        young_diabetes = 0
        young_dyslipidemia = 0
        young_hypertension = 0
        medication_gluc = 0
        medication_chol = 0
        medication_bp = 0
        mace_risk = 0
        triple_condition = 0
        counter_eightteen = 0
        counter_twentyfive = 0
        counter_sixtyfive = 0
        overweightcounter_twelve = 0
        overweightcounter_eightteen = 0
        overweightcounter_twentyfive = 0
        overweightcounter_sixtyfive = 0
        overweightcounter_ethnicity = {"Dutch": 0,
                                "Moroccan": 0,
                                "Turkish": 0,
                                "Hindustan": 0,
                                "Other": 0}
        counter_ethnicity = {"Dutch": 0,
                                "Moroccan": 0,
                                "Turkish": 0,
                                "Hindustan": 0,
                                "Other": 0}
        avg_mace_risk = []
        low_mace_risk_age = []
        low_mace_risk_ethnicity = []
        mid_mace_risk_age = []
        mid_mace_risk_ethnicity = []
        high_mace_risk_age = []
        high_mace_risk_ethnicity = []
        bmis = []

        poles = (18, 25, 65, 112)
        poles = (18, 25, 45, 55, 65, 75, 85, 95, 112)
        
        people_ages = [0] * len(poles)
        
        for agent in self.schedule.agents:
            if self.agent_condition(agent) or (agent.MACE_risk is None):
                continue
            
            avg_mace_risk.append(agent.MACE_risk)
            
            for pole in range(len(poles)):
                if agent.age < poles[pole]:
                    people_ages[pole] += 1
                    break
            
            if (agent.age < 18) and (agent.age >= 12):
                young_people += 1
                young_smoking += agent.smoking
                former_young_smoking += agent.former_smoking
                
                triple_temp = 0
                if agent.gluc >= 7:
                    young_diabetes += 1
                    triple_temp += 1
                if agent.chol > 5:
                    young_dyslipidemia += 1
                    triple_temp += 1
                if agent.syst > 140:
                    young_hypertension += 1
                    triple_temp += 1
                if agent.bmi >= 25:
                    overweightcounter_twelve += 1
                    
                if triple_temp == 3:
                    triple_condition += 1
                
            elif agent.age >= 18:
                former_adult_smoking += agent.former_smoking
                smoking += agent.smoking
                adults += 1
                if (agent.MACE_risk >= 0.20):
                    mace_risk += 1
                if (agent.MACE_risk  < 0.1):
                    low_mace_risk_age.append(agent.age)
                    low_mace_risk_ethnicity.append(agent.ethnicity)
                elif (agent.MACE_risk  >= 0.1) and (agent.MACE_risk < 0.15):
                    mid_mace_risk_age.append(agent.age)
                    mid_mace_risk_ethnicity.append(agent.ethnicity)
                else:
                    high_mace_risk_age.append(agent.age)
                    high_mace_risk_ethnicity.append(agent.ethnicity)
                    
                if agent.gluc >= 7:
                    diabetes += 1
                if agent.chol  < 5.15:
                    dyslipidemia += 1
                if agent.syst > 140:
                    hypertension += 1
                
            if (agent.age >= 18) and (agent.age <= 24):
                if agent.bmi >= 25:
                    overweightcounter_eightteen += 1
                counter_eightteen += 1
            elif (agent.age >= 25) and (agent.age <= 65):
                if agent.bmi >= 25:
                    overweightcounter_twentyfive += 1
                counter_twentyfive += 1
            elif (agent.age > 65):
                if agent.bmi >= 25:
                    overweightcounter_sixtyfive += 1                        
                counter_sixtyfive += 1
                
                bmis.append(agent.bmi)
            
            if agent.bmi >= 25:
                overweightcounter_ethnicity[agent.ethnicity] += 1                        
            counter_ethnicity[agent.ethnicity] += 1
            
            medication_gluc += agent.medication_gluc
            medication_bp += agent.medication_bp
            medication_chol += agent.medication_chol
            
        low_mace_risk_ethnicity = Counter(low_mace_risk_ethnicity)
        mid_mace_risk_ethnicity = Counter(mid_mace_risk_ethnicity)
        high_mace_risk_ethnicity = Counter(high_mace_risk_ethnicity)
        
        sorted_mace_events = sorted(self.mace_events)

        i = 0
        
        # 0: < 18, 1: [18, 24], 2: [25, 64], 3: 65+
        ages_mace = [0] * len(poles)
        
        if len(sorted_mace_events) > 0:
            # set the right pole
            for pole in poles:
                if sorted_mace_events[0] >= pole:
                    i += 1
                    
            # count per age bracket
            for age in sorted_mace_events:
                if age >= poles[i]:
                    i += 1
                ages_mace[i] += 1
                
        temp_dict = {}
        for i in range(len(ages_mace)):
            temp_dict["ages maces " + str(i)] = output_division(ages_mace[i] * 100,people_ages[i] + ages_mace[i])           
                
        dict = {"Year": float(self.schedule.steps // 12),
                                                    "Adults smoking":  smoking / adults * 100,
                                                    #"Adult former smoking": former_adult_smoking / adults * 100,
                                                    "Youngsters smoking": young_smoking / young_people * 100,
                                                    #"Youngsters former smoking": former_young_smoking / young_people * 100,
                                                    "Total medication gluc": medication_gluc/(adults + young_people) * 100,
                                                    "Total medication bp": medication_bp/(adults + young_people) * 100,
                                                    "Total medication chol": medication_chol/(adults + young_people) * 100,
                                                    "Hypertension adults":  hypertension/adults * 100,
                                                    "Diabetes adults": diabetes/adults * 100,
                                                    "Dyslipidemia adults": dyslipidemia/adults * 100,
                                                    "Hypertension youngsters":  young_hypertension/young_people * 100,
                                                    "Diabetes youngsters": young_diabetes/young_people * 100,
                                                    "Dyslipidemia youngsters": young_dyslipidemia/young_people * 100,
                                                    "Mace risk": mace_risk/adults * 100,
                                                    "All conditions": float(triple_condition),
                                                    "overweight Moroccan": output_division(overweightcounter_ethnicity["Moroccan"], counter_ethnicity["Moroccan"]),
                                                    "overweight Turkish": output_division(overweightcounter_ethnicity["Turkish"], counter_ethnicity["Turkish"]),
                                                    "overweight Hindustan": output_division(overweightcounter_ethnicity["Hindustan"], counter_ethnicity["Hindustan"]),
                                                    "overweight Dutch": output_division(overweightcounter_ethnicity["Dutch"], counter_ethnicity["Dutch"]),
                                                    "overweight Other": output_division(overweightcounter_ethnicity["Other"], counter_ethnicity["Other"]),                                   
                                                    "overweight 18-24": output_division(overweightcounter_eightteen * 100, counter_eightteen),
                                                    "overweight 25-64": output_division(overweightcounter_twentyfive * 100,counter_twentyfive),
                                                    "overweight 65+": output_division(overweightcounter_sixtyfive  * 100,counter_sixtyfive),
                                                    "overweight 12-17": output_division(overweightcounter_twelve * 100, young_people),
                                                    "MACE events": float(len(self.mace_events)),
                                                    "avg age MACE": output_division(sum(self.mace_events), len(self.mace_events)),
                                                    "low MACE risk Dutch": float(low_mace_risk_ethnicity.get("Dutch", 0)),
                                                    "low MACE risk Turkish": float(low_mace_risk_ethnicity.get("Turkish", 0)),
                                                    "low MACE risk Moroccan": float(low_mace_risk_ethnicity.get("Moroccan", 0)),
                                                    "low MACE risk Hindustan": float(low_mace_risk_ethnicity.get("Hindustan", 0)),
                                                    "low MACE risk Other": float(low_mace_risk_ethnicity.get("Other", 0)),
                                                    "mid MACE risk Dutch": float(mid_mace_risk_ethnicity.get("Dutch", 0)),
                                                    "mid MACE risk Turkish": float(mid_mace_risk_ethnicity.get("Turkish", 0)),
                                                    "mid MACE risk Moroccan": float(mid_mace_risk_ethnicity.get("Moroccan", 0)),
                                                    "mid MACE risk Hindustan": float(mid_mace_risk_ethnicity.get("Hindustan", 0)),
                                                    "mid MACE risk Other": float(mid_mace_risk_ethnicity.get("Other", 0)),
                                                    "high MACE risk Dutch": float(high_mace_risk_ethnicity.get("Dutch", 0)),
                                                    "high MACE risk Turkish": float(high_mace_risk_ethnicity.get("Turkish", 0)),
                                                    "high MACE risk Moroccan": float(high_mace_risk_ethnicity.get("Moroccan", 0)),
                                                    "high MACE risk Hindustan": float(high_mace_risk_ethnicity.get("Hindustan", 0)),
                                                    "high MACE risk Other": float(high_mace_risk_ethnicity.get("Other", 0)),
                                                    "low MACE risk ethnicity": low_mace_risk_ethnicity.most_common(5),
                                                    "mid MACE risk ethnicity": low_mace_risk_ethnicity.most_common(5),
                                                    "high MACE risk ethnicity": high_mace_risk_ethnicity.most_common(5),
                                                    "low MACE risk mean age": mean(low_mace_risk_age),
                                                    "mid MACE risk mean age": mean(mid_mace_risk_age),
                                                    "high MACE risk mean age": mean(high_mace_risk_age),
                                                    "Amount 65+": float(counter_sixtyfive),
                                                    "Amount 25-65": float(counter_twentyfive),
                                                    "Amount 18-24": float(counter_eightteen),
                                                    "Amount youngsters": float(young_people),
                                                    "Amount total":  float(counter_sixtyfive) + float(counter_twentyfive) + float(counter_eightteen) + float(young_people),
                                                    "average MACE risk": mean(avg_mace_risk),
                                                    "Average bmi": mean(bmis)
                                                    }

        self.dataCollector[self.schedule.steps + 1] = {**dict, **temp_dict}

    def export_datacollection(self, name):
        """Method to save datacollection in pickled format

        Args:
            name (str): destination of file
        """
        df = pd.DataFrame.from_dict(self.dataCollector)
        df = df[1].apply(pd.Series)
        print(df)
        pd.to_pickle(df, str(name) + ".pkl")  
            
    def get_datacollection(self):
        """Method that returns the datacollection in df

        Returns:
            DataFrame: Data collected by custom datacollector
        """
        df = pd.DataFrame.from_dict(self.dataCollector)
        df = df[1].apply(pd.Series)
        return df
        
    def stats(self):
        """Method that shows stats of model. Used for verification.
        """
        women = []
        counter_eightteen = 0
        counter_twentyfive = 0
        counter_sixtyfive = 0
        overweightcounter_eightteen = 0
        overweightcounter_twentyfive = 0
        overweightcounter_sixtyfive = 0
        ages = []
        smoking = 0
        former_smoking = 0
        young_smoking = 0
        young_people = 0

        for agent in self.schedule.agents:
            if self.agent_condition(agent):
                continue
            
            if (agent.age < 18) and (agent.age >= 12):
                young_people += 1
                if (agent.smoking == 1):
                    young_smoking += 1



            if agent.age >= 18 and agent.age <= 24:

                if agent.former_smoking == 1:
                    former_smoking += 1
                if agent.smoking == 1:
                    smoking += 1
                if agent.bmi >= 25:
                    overweightcounter_eightteen += 1
                counter_eightteen += 1
            if agent.age >= 25 and agent.age <= 64:

                if agent.former_smoking == 1:
                    former_smoking += 1
                if agent.smoking == 1:
                    smoking += 1
                if agent.bmi >= 25:
                    overweightcounter_twentyfive += 1
                counter_twentyfive += 1
            if agent.age >= 65 :

                if agent.former_smoking == 1:
                    former_smoking += 1
                if agent.smoking == 1:
                    smoking += 1
                ages.append(agent.age)
                if agent.bmi >= 25:
                    overweightcounter_sixtyfive += 1                        
                counter_sixtyfive += 1
            women.append(int(agent.bmi))   

        print( (overweightcounter_eightteen + overweightcounter_twentyfive + overweightcounter_sixtyfive)/(counter_eightteen + counter_twentyfive + counter_sixtyfive ) * 100, "procent")
        print( "18-24 ", overweightcounter_eightteen/counter_eightteen * 100, "procent", counter_eightteen, " totaal")
        print( "25-64 ", overweightcounter_twentyfive/counter_twentyfive * 100, "procent", counter_twentyfive, " totaal")  
        print( "65+ ", overweightcounter_sixtyfive/counter_sixtyfive * 100, "procent", counter_sixtyfive, " totaal")
        print("Smoking", smoking/(counter_eightteen + counter_twentyfive + counter_sixtyfive ) * 100)
        print("Former smoking", former_smoking/(counter_eightteen + counter_twentyfive + counter_sixtyfive ) * 100)
        print("Young smoking", young_smoking/young_people * 100, young_smoking, "young people", young_people)

        print(Counter(ages))           
        # plt.hist(women, edgecolor="red", bins=20)
        # plt.show()
        women = Counter(women)
        print((counter_eightteen + counter_twentyfive + counter_sixtyfive ), women.most_common())
            
    def __hash__(self):
        """Method to define iteration order

        Returns:
            6: to set iteration order
        """
        return 6