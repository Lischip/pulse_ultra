from cmath import inf
from mesa import Agent
import numpy as np
from enum import Enum
import numpy as np
import math
from collections import Counter


alpha = np.repeat([-31.1, 485.9, 692.6, 486.6, 835.6, 658.5], [2, 7, 7, 11, 30, 60])
beta = np.repeat([57.317, 20.315, 13.384, 14.818, 8.126, 9.082], [2, 7, 7, 11, 30, 60])

mean_friends = np.repeat([0, 5.01, 8.01, 10.82, 11.39, 11.13, 10.01, 9.36, 9.17, 8.28, 6.76, 6.14], [12, 4, 4, 6, 5, 5, 10, 5, 5, 14, 9, 32])
std_friends = np.repeat([0, 4.04, 9.37,  9.87, 9.57, 9.09, 8.62, 8.08, 7.70, 6.94, 5.86,5.38], [12, 4, 4, 6, 5, 5, 10, 5, 5, 14, 9, 32]) 

expenditure_multipliers = [1.2, 1.375, 1.55, 1.725, 1.9]

SEE = np.repeat([59, 70, 111, 119, 111, 108], [4, 8, 8, 10, 30, 50])
  
cholesterol = [0.17219137187535, -0.00010863665945, 0.00051339500368, 0.02569411851744, 0.02296041232569, 0.00478613143738, 0.01155159178380, 0.00337794030042]
cholesterol_std_error = [0.00091538507896, 0.00000885858858, 0.00002112475230, 0.00025091090573, 0.00070568255857, 0.00030540248793, 0.00040008461583, 0.00064006894661] #* 124733
cholesterol_std_error = [x * np.sqrt(124733) for x in cholesterol_std_error]

blood_pressure = [4.61548614246050, 0.00328166754506, 0.00362201068541837, 0.0422593390422044, -0.02974207430813, -0.00468999373620, -0.00986526955536, -0.03040992814217 ]
blood_pressure_std_error = [0.00231674842132, 0.00002213968379, 0.0000587771020721322, 0.00238728845631174, 0.00180441109810, 0.00087991566221, 0.00111476412143, 0.00179317466507] #* 151672
blood_pressure_std_error = [x * np.sqrt(151672) for x in blood_pressure_std_error]

gluc_fasting = [0.20616044226754, -0.00009973456996, -0.00067240224004, -0.05020920312190, -0.00097843287479, 0.00003660654806, 0.00168332741824, 0.00222240376698]
gluc_fasting_std_error = [0.00075144200173, 0.00000730567989, 0.00001673539932, 0.00021997118711, 0.00048054136533, 0.00025207667179, 0.00031271277534, 0.00049592383847] #* 131328
gluc_fasting_std_error = [x * np.sqrt(131328) for x in gluc_fasting_std_error]

breslow = np.load("D:/juju/lumc/pulse_ultra/input/breslow.npy")

fine_and_gray = np.array([0.0889104888756875, 0.014041347609817, 0.0663937915057769, \
    -0.0143795204966108, 1.3571621432417, 0.508149039892725, 0.612456746581127, \
        1.1507854793352, 1.43648085115873, -0.257561430584195, \
            -0.240799376082464, -0.000171793446414203, -0.000893179412407768, \
                -0.000870519026879657, -0.0144884589032191, -0.00306390553304349, \
                    -0.00445550621721679, -0.0128246003020665, -0.0181564617236253])

def gamma_linkinv(x):
    """Returns inverse of the link function in gamma distribution

    Args:
        x (float): predictor that needs to be transformed

    Returns:
        float: value of the gamma inverse of the link function in gamma distribution
    """
    return 1/x

def gaussian_linkinv(x):
    """Returns inverse of the link function in gaussian distribution

    Args:
        x (float): predictor that needs to be transformed

    Returns:
        float: value of the gamma inverse of the link function in gaussian distribution
    """
    return np.exp(x)


class zlist(list):
    """Custom datatype. List that never exceeds 12 items.

    Args:
        list (list): The list of 12 items
    """
    def append(self, item):
        list.append(self, item)
        if len(self) > 12: self.clear()
        
class MaceRiskDict(dict):
    """Custom dict that keeps track of the risk of the first-ever MACE in
    10-years. 
    """
    def __init__(self, val=None):
        if val is None:
            val = {}
        super().__init__(val)
        self.default = -1
        
    def set_default(self, value):
        if self.default == -1:
            self.default = value

    def __setitem__(self, key, value):
        super().__setitem__(key + 10, value)

        if super().__len__() > 11:
            super().__delitem__([key for key in super().keys()][0])
            self.default = [key for key in super().keys()][0]

    def __delitem__(self, key):
        super().__delitem__(key)
        
    def __missing__(self, key):
        return self.default


class Ethnicity_multiplier(Enum):
    """Enum that keeps track of the value the 10-year first-ever MACE risk needs
    to be multiplied with based on ehtnicity.
    """
    Dutch = 1
    Hindustan = 1.63
    Other = 0.71
    Turkish = 0.87
    Moroccan = 0.82


class MaceAgent(Agent):
    """An agent"""
    
    class Gamma(Enum):
        """Enum for average weight gain when growing one cm in height
        """
        Dutch = 0.516
        Turkish = 0.516
        Moroccan = 0.516
        Hindustan = 0.469
        Other = 0.516

    class Height(Enum):
        """Enum to keep track of height formulas
        """
        Dutch = [0.001855,  0.886,  65.57, 171.04208, 170.7, 6.3]
        Turkish = [0.001966,  0.8738, 65.43, 160.778736, 161, 6.4]
        Moroccan = [0.001927, 0.8737, 65.11, 162.91019200000002, 162.8, 6.5]
        Hindustan = [0.002095, 0.9032, 63.77, 158.33552, 159.6, 5.9]
        Other = [0.001855,  0.886,  65.57, 171.04208, 161, 6.2]
        
    class Husband_height(Enum):
        """Enum that generates a height for the husband(or partner) of an agent
        """
        Dutch = [183.8, 7.1]
        Turkish = [177.5, 6.8]
        Moroccan = [177.6, 7.6]
        Hindustan = [174.5, 6.9]
        Other = [183.8, 7.1] 
        
    class Homogeneity(Enum):
        """Enum of the average fraction of homogeneity of the in the friend group 
        of a woman of a certain ethnicity
        """
        Dutch = 0.88412
        Turkish = 0.54503
        Moroccan = 0.45198
        Hindustan = 0.14899
        Other = 0.21829

    class Dutch(Enum):
        """Enum of the average fraction of Dutch people in in the friend group 
        of a woman of a certain ethnicity
        """
        Turkish = 0.35
        Moroccan = 0.37
        Hindustan = 0.87
        Other = 0.75
    
    @property
    def metabolic_rate(self):
        """Method to calculate the metabolic rate

        Returns:
            float: metabolic rate
        """
        return self.expenditure * ((alpha[self.age] + (beta[self.age] * self.weight)) + self.SEE * SEE[self.age])/0.9
      
    @property
    def categorised_ethnicity(self):
        """Used to calculate the 10-year first-ever MACE risk. Has no other
        purpose. Essentially maps the ethnicities to binaries and puts this in
        a list

        Returns:
            list: list unique to each ethnicity
        """
        ethnicity_array = [0] * 4
        ethnicities = [ "Morrocan", "Other", "Hindustan", "Turkish"]
        for ethnicity in range(len(ethnicities)):
            if self.ethnicity == ethnicities[ethnicity]:
                ethnicity_array[ethnicity] = 1
        return ethnicity_array
    
    def get_med_usage(self):
        """method to determine medication usage of agent
        """
        if (self.medication_gluc == 0):
            if (self.gluc >= 7) or (self.gluc_value(1) >= 7.78): #old: 54, 58
                self.medication_gluc = 1
            
        if (self.medication_bp == 0):
            if (self.syst >= 130) or self.syst_value(1) >= 140:
                self.medication_bp = 1
                            
        if (self.medication_chol == 0):
            if (self.chol < 5.33):
                self.medication_chol = 1
   
    def get_intention(self):
        """Method that calculates, based on BMI, the intention to lose weight
        (due to external factors)

        Returns:
            list: months the agents will be dieting
        """
        # Perception of weight status and dieting behaviour in Dutch men and women
        if (self.bmi <25):
            weights = [0.682, 0.161, 0.092, 0.066]
        elif(self.bmi < 30):
            if not self.gluc < 7:
                weights = [0.34, 0.12, 0.155, 0.174]
            else:
                weights = [0.437, 0.221, 0.15, 0.193]
        else:
            weights = [0.319, 0.165, 0.327, 19]
        nr_months = self.random.choices([0, self.random.randint(1,2), self.random.randint(3,5), self.random.randint(6, 10)], weights)[0]
        sample = self.random.sample(range(0, 11), k = nr_months)
        if len(sample) > 1:
            sample.sort()
        return sample
        
    def get_smoking_temptation(self):
        """Function that calculates smoking temptation from external factors

        Returns:
            float: temptation
        """
        income_multiplier = [1.4, 1.25, 1]
        if (self.age >= 12) and (self.age < 18):
            value = 0.2
            return value * income_multiplier[self.income]  * 1.5 * self.model.smoking_trend[self.model.schedule.steps]
        else:
            value = 0.1
        return value * income_multiplier[self.income]
    
    def dis_value(self, array, linkinv, medication = 0):
        """Method that returns the value for fasting blood glucose,
        systolic blood pressure, or total cholesterol

        Args:
            array (list): array corresponding to values for fasting blood glucose,
            systolic blood pressure, or total cholesterol
            linkinv (method): The linkinverse corresponding to the GLM
            medication (int, optional): whether or not the agent is using medication.
            Defaults to 0.

        Returns:
            float: Value for blood glucose, systolic blood pressure, or total 
            cholesterol given the array (make sure linkinv matches up)
        """
        ethnicity_array = self.categorised_ethnicity

        x = array[0] + (self.age * array[1]) + (self.bmi * array[2]) +  \
            (medication * array[3]) + (ethnicity_array[0] * array[4]) + \
                (ethnicity_array[1] * array[5]) + (ethnicity_array[2] * array[6]) + \
                    (ethnicity_array[3] * array[7])
                    
        return linkinv(x)
    
    def chol_value(self, medication = 0):
        """Returns total cholesterol value

        Args:
            medication (int, optional): whether the agent uses medication or not.
            Defaults to 0.

        Returns:
            float: total cholesterol value
        """
        return self.dis_value(cholesterol, gamma_linkinv, medication)
     
    def syst_value(self, medication = 0):
        """Returns systolic blood pressure value. Includes effect of pregnancy
        complications.

        Args:
            medication (int, optional): whether the agent uses medication or not.
            Defaults to 0.

        Returns:
            _type_: systolic blood pressure value
        """
        value =  self.dis_value(blood_pressure, gaussian_linkinv, medication)
        if (self.pregnancy_complications["hypertension"] < self.model.schedule.steps):
                return max(value, 140)
        return value
     
    def gluc_value(self, medication = 0):
        """Returns fasting blood glucose value. Includes effect of pregnancy
        complications.

        Args:
            medication (int, optional): whether the agent uses medication or not.
            Defaults to 0.

        Returns:
            _type_: fasting blood glucose value
        """
        value = self.dis_value(gluc_fasting, gamma_linkinv, medication)
        if (self.pregnancy_complications["diabetes"] < self.model.schedule.steps):
            return max(7, value)
        return value

    
    @property
    def MACE_risk(self):
        """Returns the 10-year first-ever MACE risk of the agent

        Returns:
            float: 10-year first-ever MACE risk
        """
        x = np.array([self.age, self.syst, self.gluc, self.chol, \
            self.smoking, self.former_smoking, self.medication_gluc, \
                self.medication_bp, self.medication_chol, (1 if self.income == 1 else 0), \
                    (1 if self.income == 2 else 0), self.age * self.syst, \
                        self.age * self.gluc, self.age * self.chol, \
                            self.age * self.smoking, self.age * self.former_smoking, \
                                self.age * self.medication_gluc, self.age * self.medication_bp, \
                                    self.age * self.medication_chol])
        
        x = np.multiply(x, fine_and_gray)
        x = sum(x)
        return 1-np.exp(-np.cumsum(breslow * np.exp(x)))[119] #119 is 10 years
    
    def MACE_risk_age(self, age):
        """Returns the 10-year first-ever MACE risk of the agent at a certain
        age

        Args:
            age (int): age (in years)

        Returns:
            float: 10-year first-ever MACE risk
        """
        x = np.array([age, self.syst, self.gluc, self.chol, \
            self.smoking, self.former_smoking, self.medication_gluc, \
                self.medication_bp, self.medication_chol, (1 if self.income == 1 else 0), \
                    (1 if self.income == 2 else 0), age * self.syst, \
                        age * self.gluc, age * self.chol, \
                            age * self.smoking, age * self.former_smoking, \
                                age * self.medication_gluc, age * self.medication_bp, \
                                    age * self.medication_chol])
        
        x = np.multiply(x, fine_and_gray)
        x = sum(x)
        
        return 1-np.exp(-np.cumsum(breslow * np.exp(x)))[119]
    
    def height_formula(self):
        """Calculates height per step for agent

        Returns:
            float: height
        """
        return -(self.Height[self.ethnicity].value[3] - self.target_height) -self.Height[self.ethnicity].value[0] * self.age_in_months**2 + self.Height[self.ethnicity].value[1] * self.age_in_months + self.Height[self.ethnicity].value[2]

    def __init__(self, unique_id, model, ethnicity, age, height, income):
        """Initialisatoin

        Args:
            unique_id (int): unique id of agent
            model (MaceModel): model that the agent resides in
            ethnicity (str): ethnicity of the agent
            age (int): age of the agent (in years)
            height (float): height of the agent (current height or target height)
            income (int): income category (0, 1 or 2)
        """
        super().__init__(unique_id, model)
        
        # immigrants
        self.activation = -1
        self.health = 1
        self.ethnicity = ethnicity
        self.age = age
        self.death_days = -1
        self.income = income

        self.dead = False
        
        self.rng = model.rng
        
        self.medication_chol = 0
        self.medication_bp = 0
        self.medication_gluc = 0

        self.smoking = 0
        self.former_smoking = 0
        
        self.pregnancy_complications = {"hypertension": inf,
                                        "diabetes":  inf}

        if self.age >= 18:
            self.smoking_threshold = self.random.uniform(0.32, 1.3)
        else:
            self.smoking_threshold = self.random.uniform(self.model.smoking_starting_rage[0], self.model.smoking_starting_rage[1])      
        self.smoking_settings()
        self.smoking_desire = 0
        self.weight = None
        self.intention = []
        
        self.SEE = self.random.triangular()
        
        # old people:
        self.unintended_wl = False
        
        # for epsilon
        self.epsilon = self.rng.normal(loc=7716, scale=1000)
        
        self.birthday_month = self.random.choice(range(0, 11))
        if age < 21:
            self.target_height = height
            self.height = self.height_formula()
        else:
            self.height = height
        amount_friends_mult = self.random.uniform(-1, 1)
        self.friends_over_life= np.ones([111])
        self.number_children = 0
        self.mother = None
        self.husband_height = 0
        
        self.tolerance_of_diff = self.random.uniform(0.04, 0.1)

        self.dieting = -2
        self.addicted = False
        
        self.smoking_friends = self.random.uniform(0, 0.1)
        self.external_smoking_push_factor = self.random.uniform(0, 0.1) 
        self.susceptibility  = self.random.uniform(model.susceptibility[0], model.susceptibility[1])

        self.macerisks = MaceRiskDict()
        
        self.pregnant = False
        self.pregnancy_countdown = inf

        for i in range(len(self.friends_over_life)):
            self.friends_over_life[i] = max( 1, math.ceil(mean_friends[i] + amount_friends_mult*std_friends[i]))
 
        self.friends_over_life = self.friends_over_life.astype(int)
        
        self.set_friends_eth()
        self.nodes = []
        self.degrees = {}
        
        ## interventions
        self.school_months = 0
      
    def smoking_settings(self):
        """Function to initialise smoking for agents aged between 12 and 18
        """
        if (self.age >= 12) and (self.age < 18):
            smoke_temp = self.random.choices([0,1,2], self.model.smoking_children)[0]
            if smoke_temp == 0:
                self.smoking = 1
                self.smoking_threshold = self.random.triangular(self.model.smoking_quitting_range[0], self.model.smoking_quitting_range[1], self.model.smoking_quitting_range[0] )
                self.smoking_desire = 0.7
            elif smoke_temp == 1:
                self.former_smoking = 1
                self.has_smoked = 1
                
    def starts_smoking(self):
        """Function that reconfigures an agent's attributes after she has
        started smoking.
        """
        self.smoking = 1
        self.smoking_threshold = self.random.triangular(self.model.smoking_quitting_range[0], self.model.smoking_quitting_range[1], self.model.smoking_quitting_range[0] )
        self.smoking_desire = 0.7
        if (self.random.uniform(0, 1) < 0.1):
            self.addicted = True
        for friend in self.nodes:
            if (friend.smoking == 0) and (self.random.uniform(0, 1) < 0.2):
                friend.smoking = 1
                friend.smoking_threshold = self.random.triangular(self.model.smoking_quitting_range[0], self.model.smoking_quitting_range[1], self.model.smoking_quitting_range[0] )
                friend.smoking_desire = 0.7

    def mortality_risk(self, age, year):
        """Calculates mortality risk of the agent

        Args:
            age (int): Age of agent (in years)
            year (int): Current year (in years, starting from 0)

        Returns:
            float: mortality risk
        """
        return (0.000010694877 * age**3 - 0.00215675241 * age**2 +  0.145068422 * age - 3.24393183)  * np.exp(-0.0013771318811465964 * year)

    def unintended_wl_risk(self, age, year):
        """Calculates the odds of an agent suffering from unintended weight loss
        due to old age

        Args:
            age (int): Age of agent (in years)
            year (int): Current year (in years, starting from 0)

        Returns:
            float: risk to unintentionally start losing weight (due to old age)
        """
        return (5.74490920e-07 * age **3 + 3.96785538e-05 * age ** 2 +6.81189843e-04 * age -3.19688799e-01) * np.exp(-0.0013771318811465964 * year)    
    
    def set_friends_eth(self):
        """Ethnicitiy distribution of friends the agent wants
        """
        #0: homogeneity, 1: Dutch, 2: other
        if (self.ethnicity != "Dutch"):
            friends_eth = Counter(self.model.random.choices(range(0, 3), weights=[self.Homogeneity[self.ethnicity].value, self.Dutch[self.ethnicity].value, 1 - self.Dutch[self.ethnicity].value - self.Homogeneity[self.ethnicity].value] , k=self.amount_friends))
        else:
            friends_eth = Counter(self.model.random.choices([0,2], weights=[self.Homogeneity[self.ethnicity].value, 1 - self.Homogeneity[self.ethnicity].value], k=self.amount_friends))
        self.friends_eth = dict([(0, 0), (1,0), (2, 0)])
        self.friends_eth.update(dict(friends_eth))
        
    def energy_out(self, expenditure, intake):
        """Method that calculates the caloric expenditure

        Args:
            expenditure (float): energy expenditure multiplier
            intake (float): energy intake
        
        Returns:
            float: caloric expenditure
        """
        return expenditure * (alpha[self.age] + (beta[self.age] * self.weight) + self.SEE * SEE[self.age]) + 0.1 * intake

    def caloric_change(self, calories):
        """Function that caclulates the healthy change in calories. Prevents
        agents from becoming underweight.

        Args:
            calories (float): caloric increase (or decrease)
        """
        metabolic_rate = self.expenditure * ((alpha[self.age] + (beta[self.age] * self.weight)) + self.SEE * SEE[self.age])/0.9
        if self.random.uniform(0, 1) < self.susceptibility :
            self.caloric_intake = max(metabolic_rate, self.caloric_intake + calories)  
            
    def school_intervention(self, fraction, months):
        """Method that is called by the model class to enact school intervention

        Args:
            fraction (float): Change in susceptibility
            months (list): The months the policy will be active
        """
        self.susceptibility  = max(0, self.susceptibility  * fraction)
        
        if self.bmi >= 25:
            self.school_months = months
            
    def school_intervention_weight_loss(self):
        """Method that is called by this class to process weight loss due to
        the school intervention
        """
        self.weight = self.weight - 2
        metabolic_rate = self.expenditure * ((alpha[self.age] + (beta[self.age] * self.weight)) + self.SEE * SEE[self.age])/0.9
        self.school_caloric_intake = metabolic_rate

    def change_external_smoking_push_factor (self, value, months):
        """Method that is called by the model for the smoking intervention

        Args:
            value (tuple): tuple that determines the new lower and upper limit
            of the factor
            months (list): Months the intervention will be active 
        """
        if hasattr(self, "smoking_past"):
            return
        if (self.smoking):
            self.smoking_past = [self.external_smoking_push_factor , self.model.schedule.steps + months]
            self.external_smoking_push_factor = self.random.uniform(value[0], value[1])
        
    def weight_loss(self):
        """Function that takes care of weight loss of an agent that is suffering
        from unintended weight loss
        """
        self.susceptibility  = 0
        self.unintended_wl = self.model.schedule.steps + self.random.uniform(0, 11)
        self.weight_to_lose = self.random.uniform(1,4)
    
    def weight_change(self, energy_in, energy_out):
        """Method that calculates weight change

        Args:
            energy_in (float): caloric intake
            energy_out (float): caloric expenditure
        """
        if (self.unintended_wl):
            if (self.unintended_wl > self.model.schedule.steps):
                if self.bmi > 20:
                    self.weight -= self.weight_to_lose
                return
                
        self.weight = self.weight + ((self.newHeight - self.height) * self.Gamma[self.ethnicity].value) + ((energy_in - energy_out)/self.epsilon)
        self.height = self.newHeight
        
    def set_bmi(self, bmi):
        """An agents bmi can be configured with this function. This will affect
        her expenditure and caloric intake.

        Args:
            bmi (float): desired BMI
        """
        if (self.age >= 12):
            self.weight = bmi * (self.height/100)**2
        else:
            self.target_bmi = bmi
            return

        if self.age >= 12:
            #copy from momma
            if self.mother:
                self.expenditure = self.mother.expenditure
            else:
                self.expenditure = self.model.random.choices(expenditure_multipliers, self.model.exercise_amount(self.age))[0]
        else:
            self.expenditure = expenditure_multipliers[0]
        self.caloric_intake = self.expenditure * ((alpha[self.age] + (beta[self.age] * self.weight)) + self.SEE * SEE[self.age])/0.9
        
        self.chol = self.chol_value()
        self.syst = self.syst_value()
        self.gluc = self.gluc_value()
 
        self.get_med_usage()

    def tolerate(self, other):
        """Method that can be called to see if the agent would want to befriend
        someone

        Args:
            other (MaceAgent): The candidate the agent is considering befriending

        Returns:
            boolean: True if the agents can be friends, False if not.
        """
        difference = abs(self.bmi - other.bmi)
        if self.random.uniform(0, 1) < (-self.tolerance_of_diff * difference + 1):
            return True
        return False

    @property
    def amount_friends(self):
        """Return the number of friends the agent wants right now (with her
        current age)

        Returns:
            int: the amount of friends the agent wants now
        """
        return self.friends_over_life[self.age] 
    
    @property
    def age_in_months(self):
        """Calculates the agent's age in months

        Returns:
            int: The agent's age in months
        """
        current_month = self.model.schedule.steps % 12
        if current_month < self.birthday_month:
            return (12 - current_month) + self.age * 12
        else:
            return (current_month - self.birthday_month) + self.age * 12

    @property 
    def bmi(self):
        """Returns the current BMI of the agent

        Returns:
            float: current BMI
        """
        if self.weight == None:
            return None
        else:
            return self.weight/((self.height/100)**2)
    
    @property
    def current_friends(self):
        """Ethnicity distribution of friends the agent actually has.
        """
        current_friends = dict([(0, 0), (1,0), (2, 0)])
        
        for friend in self.nodes:  
            if friend.ethnicity == self.ethnicity:
                current_friends[0] += 1
            elif friend.ethnicity == "Dutch": # won't be triggered if agent is Dutch and friend is Dutch
                current_friends[1] += 1
            else:
                current_friends[2] += 1
                        
        return current_friends   
    
    def birth(self):
        """Method that should be called when an agent gives birth. Adds new
        agent to model.
        """
        if self.husband_height == 0:
            self.husband_height = self.model.rng.normal(loc=self.Husband_height[self.ethnicity].value[0], scale=self.Husband_height[self.ethnicity].value[1])
        
        target_height = 47.1 + 0.335 * self.husband_height + 0.364 * self.height
        
        target_height_sd = abs((target_height - self.Height[self.ethnicity].value[4])/self.Height[self.ethnicity].value[5])
        target_height = self.random.uniform(target_height - target_height_sd, target_height + target_height_sd)
        a= MaceAgent(self.model.id, self.model, self.ethnicity, 0, height = target_height, income = self.income)
        a.mother = self
        self.model.id += 1
        self.model.agents.append(a)
        self.model.schedule.add(a)         
        
    def step(self):
        """Advance agent one step.
        """
        if (self.dead):
             return
         
        if (self.activation > self.model.schedule.steps):
            return
        elif (self.activation == self.model.schedule.steps):
            self.activation = -1
        


        #birthday
        if self.model.schedule.steps % 12 == self.birthday_month:
            self.age += 1
            if (self.age == 110):
                self.model.deathkeeper[self.model.schedule.steps//12] = self.model.deathkeeper.get(self.model.schedule.steps//12, 0) + 1
                self.dead = True
                self.model.deathlist.add(self)
                return
                       
            self.set_friends_eth()
            # initialise new agent
            if self.age == 12:
                self.newHeight = self.height_formula()
                self.height = self.newHeight
                if self.mother != None: # if u have a mom, copy, if not use placeholder.
                    self.set_bmi(self.mother.bmi)
                    if self.mother.smoking == 1:
                        self.starts_smoking()
                else:
                    self.set_bmi(self.target_bmi)
                    self.smoking_settings()           
                
            if (self.model.schedule.steps == 0):
                pass
            elif (self.former_smoking == 0) and (self.age >= 12) and (self.age <= 65):
                if (self.smoking == 0) and (self.smoking_desire > self.smoking_threshold):
                    self.starts_smoking()
                elif (self.age >= 12) and (self.age <= 65) and (self.smoking == 1) and not(self.addicted):
                    if (self.smoking_desire < self.smoking_threshold):
                        self.smoking = 0
                        self.former_smoking = 1
                
            # Will agent start smoking this year?
            elif (self.age >= 12) and (self.age <= 65) and (self.smoking == 0) and (self.former_smoking == 0) and (self.random.uniform(0, 1) < self.get_smoking_temptation()):
                self.starts_smoking()
                     
        if (self.model.schedule.steps % 12 == 0) and (self.age >= 75):
            # Will agent die this year?
            # deathday
            multiplier = 1
            if (self.bmi >= 30):
                multiplier = 1.10 
            if not (self.dead) and not (self.pregnant) and (self.model.random.uniform(0, 1) < min(1, self.mortality_risk(self.age, self.model.schedule.steps //12) * multiplier)):
                self.death_days = self.model.schedule.steps + self.random.randrange(0, 11)
        
            # Will agent lose unintentionally loose weight?
            if (not self.unintended_wl) and \
                (self.random.uniform(0, 1) < self.unintended_wl_risk(self.age, self.model.schedule.steps //12)):
                        self.weight_loss()

        # add to deathlist if agent dies         
        if self.model.schedule.steps == self.death_days :
            self.model.deathkeeper[self.model.schedule.steps//12] = self.model.deathkeeper.get(self.model.schedule.steps//12, 0) + 1
            self.model.deathlist.add(self)
            return   
                        
        if self.age < 12:
            return # we only care about agents >= 12  
        elif (self.age > 12) and (self.age < 21):    
            self.newHeight = self.height_formula()
        else:
            self.newHeight = self.height
   
        # Will agent lose weight (external factors)?
        if (self.bmi != None) and (self.model.schedule.steps % 12 == 0) and (self.age <= 45):
            self.intention = self.get_intention()

        month = self.model.schedule.steps % 12

        if (month == self.dieting + 1) and (self.bmi != None) and (self.age <= 45) and (self.number_children < 1) :
            if (self.random.uniform(0, 1) > 0.3):
                self.caloric_intake = self.old_caloric_intake
                self.dieting = -2
        elif (self.bmi != None) and (self.age <= 45) and (month in self.intention) and (self.number_children < 1):
            self.old_caloric_intake = self.caloric_intake
            metabolic_rate = self.expenditure * ((alpha[self.age] + (beta[self.age] * self.weight)) + self.SEE * SEE[self.age])/0.9
            self.caloric_intake = max(metabolic_rate, self.caloric_intake - self.random.uniform(1, 500))
            self.intention.remove(month)
            self.dieting = month
            
        # Intervention 1 "School"
        if self.school_months > 0:
            self.school_intervention_weight_loss()
            self.caloric_intake = min(self.school_caloric_intake, self.caloric_intake)
            self.school_months -= 1 
            
        # intervention 2 "Smoking"   
        if hasattr(self, 'smoking_past') and (self.model.schedule.steps == self.smoking_past[1]):
            self.external_smoking_push_factor = self.smoking_past[0]
            delattr(self, "smoking_past")
            
        energy_out = self.energy_out(self.expenditure, self.caloric_intake)     
        self.weight_change(self.caloric_intake, energy_out)
        
        
          
        if self.bmi != None:
            self.chol = self.chol_value(self.medication_chol)
            self.syst = self.syst_value(self.medication_bp)
            self.gluc = self.gluc_value(self.medication_gluc)
            # only decide yearly if they need medication or not
            if self.bmi >= 25 and (self.model.schedule.steps % 12 == self.birthday_month) :
                self.get_med_usage()
                
            if not (self.pregnant) and self.model.schedule.steps % 12 == self.birthday_month:
                self.macerisks.set_default(self.MACE_risk_age(self.age-5) * Ethnicity_multiplier[self.ethnicity].value )

                self.macerisks[self.model.schedule.steps//12] = self.MACE_risk * Ethnicity_multiplier[self.ethnicity].value
                if (self.random.uniform(0, 1) < self.macerisks[self.model.schedule.steps//12]) and \
                    (self.bmi >= 30):
                    self.model.macedeathkeeper[self.model.schedule.steps//12] =  self.model.macedeathkeeper.get(self.model.schedule.steps//12, 0) + 1
                    self.model.mace_events.append(self.age)
                    self.model.deathkeeper[self.model.schedule.steps//12] =  self.model.deathkeeper.get(self.model.schedule.steps//12, 0) + 1
                    self.dead = True
                    self.model.deathlist.add(self)
                    return
        
        if (self.pregnant):
            self.pregnancy_countdown -= 1
            if (self.pregnancy_countdown == 0):
                self.number_children += 1
                self.birth()
                #pregnancy effects
                if (self.random.uniform(0, 1) < 0.05):
                    self.pregnancy_complications["hypertension"] = self.model.schedule.steps + (5 * 12)
                if (self.random.uniform(0, 1) < 0.075):
                    self.pregnancy_complications["diabetes"] = self.model.schedule.steps + (5 * 12)

                self.pregnant = False
        
    def __hash__(self):
        """Method to define iteration order

        Returns:
            6: to set iteration order
        """
        return 6