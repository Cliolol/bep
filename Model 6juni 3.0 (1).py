#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:04:53 2024

@author: cleovos
"""

#%%

import math

import numpy as np

from enum import Enum
import networkx as nx

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
#from mesa.batchrunner import BatchRunnerMP, BatchRunnerMP, FixedBatchRunner

import seaborn as sns
import pandas as pd

import random

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc, collections
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from IPython.display import HTML


#%% Nummeren verschillende states

# nummeren van verschillende states

class State(Enum):
    NIET_BESMET = 0
    BESMET_ZONDER_VERSCHIJNSELEN = 1
    BESMET_MET_VERSCHIJNSELEN = 2
    HERSTELD = 3
    
#%% Shifts 

#shift 1 = 7u - 15u 
#shift 2 = 15u - 23u 
#shift 3 = 23u - 7u 


route_schoonmaker = [("recreatiekamer",0.8),("platform",0.8),("machinekamer",0.8),("badkamers",0.8),("toiletten",0.8),("brug",0.8),("hut",0.8),("keuken",0.8),("kantine",0.8)]

#%% Roosters

kok = [("hut", 7), ("keuken", 2), ("recreatiekamer", 1), ("keuken", 3), ("recreatiekamer", 3), ("keuken", 5), ("recreatiekamer", 1), ("hut", 2)]

machinemedewerker_1 = [("hut",7),("machinekamer",1),("kantine",1),("machinekamer",3),("kantine",1),("machinekamer",2),("recreatiekamer",3),("kantine",2),("recreatiekamer",2),("hut",2)]
machinemedewerker_2 = [("hut",8),("kantine",1),("recreatiekamer",3),("kantine",2),("machinekamer",3),("kantine",2),("machinekamer",3),("hut",1)]
machinemedewerker_3 = [("machinekamer",7),("hut",1),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",5),("kantine",2),("recreatiekamer",2),("hut",1),("machinekamer",1)]


platformmedewerker_1 = [("hut",7),("platform",1),("kantine",1),("platform",3),("kantine",1),("platform",2),("recreatiekamer",3),("kantine",2),("recreatiekamer",2),("hut",2)]
platformmedewerker_2 = [("hut",8),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatieruimte",2),("platform",3),("kantine",2),("platform",3),("hut",1)]
platformmedewerker_3 = [("platform",7),("hut",1),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",5),("kantine",2),("recreatiekamer",2),("hut",1),("platform",1)]


brugmedewerker_1 = [("hut", 7),("brug",8),("recreatiekamer", 3),("kantine",2),("recreatiekamer",2),("hut",2)]
brugmedewerker_2 = [("hut",8),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",2),("brug",8),("hut",1)]
brugmedewerker_3 = [("brug",7),("hut",1),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",5),("kantine",2),("recreatiekamer",2),("hut",1),("brug")]

schoonmaker_1 = [("hut", 7),("recreatiekamer",0.8),("platform",0.8),("machinekamer",0.8),("badkamers",0.8),("toiletten",0.8),("brug",0.8),("hut",0.8),("keuken",0.8),("kantine",0.8),("recreatiekamer", 3),("kantine",2),("recreatiekamer",2),("hut",2)]
schoonmaker_2 = [("hut",8),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",2),("recreatiekamer",0.8),("platform",0.8),("machinekamer",0.8),("badkamers",0.8),("toiletten",0.8),("brug",0.8),("hut",0.8),("keuken",0.8),("kantine",0.8),("hut",1)]
schoonmaker_3 = [("platform",0.77),("machinekamer",0.77),("badkamers",0.77),("toiletten",0.77),("brug",0.77),("hut",0.77),("keuken",0.77),("kantine",0.77),("hut",1),("kantine",1),("recreatiekamer",3),("kantine",1),("recreatiekamer",5),("kantine",2),("recreatiekamer",2),("hut",1),("recreatiekamer",1)]



#%% Model -klasse
class Virus(Model):
    "Een model van norovirus onder de bemanningsleden van een schip"

    def __init__(self, aantal_bemanningsleden):
        super().__init__()
        self.aantal_bemanningsleden = aantal_bemanningsleden
        self.running = True
        self.compact_mapping = {}        

        
        #netwerk maken 
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.aantal_bemanningsleden))
        self.grid = NetworkGrid(nx.grid_2d_graph(10, 10))
        self.schedule = RandomActivation(self)
        
        
        #Datacollecter maken 
        model_metrics = {
            "KeukenBezetting": self.get_keuken_bezetting}
        agent_metrics = {}
        
        self.datacollector = DataCollector(model_reporters = model_metrics, agent_reporters = agent_metrics)

       
        self.bemanningsleden_aanmaken()
        self.init_compartimenten()
        

    def bemanningsleden_aanmaken(self): 
        functies = ["kok", "schoonmaker", "machinemedewerker","platformmedewerker", "brugmedewerker"]
        vereiste_functies = [("kok",2), ("schoonmaker",3)]
        overige_functies = ["machinemedewerker","platformmedewerker", "brugmedewerker"]
       
        agent_id = 0
        
        #kok en schoonmakers: 
        for functie, aantal in vereiste_functies: 
            for _ in range(aantal):
                shift = random.choice([1,2,3])
                a = Bemanningslid(agent_id,self,shift, functie)
                self.schedule.add(a) 
                agent_id += 1
                # print(f'bemanningslid aangemaakt met functie {functie} en agent_id {agent_id}')
                
        #de overige bemanningsleden een functie toewijzen  
        resterende_bemanningsleden = self.aantal_bemanningsleden - 5
        for i in range(resterende_bemanningsleden): 
            functie = overige_functies[i % len(overige_functies)]
            shift = random.choice([1,2,3])
            a = Bemanningslid(agent_id, self, shift, functie)
            self.schedule.add(a)                   
            agent_id += 1
            # print(f'bemanningslid aangemaakt met functie {functie} en agent_id {agent_id}')
           
  
   
    def init_compartimenten(self):
    
        compartiment_types = ["kantine", "keuken", "brug", "machinekamer", "platform", "recreatiekamer"]
        
        #voeg dan voor ieder bemanningslid een eigen hut toe met eigen toilet
        for i in range(self.aantal_bemanningsleden):
            compartiment_types.append("hut")
            compartiment_types.append("toilet")
            
        aantal_badkamers = self.aantal_bemanningsleden // 4
        for badkamers in range(aantal_badkamers):
            compartiment_types.append("badkamer")
            
        self.plaats_clusters(compartiment_types)
        
        
    def plaats_clusters(self, compartiment_types):
        cluster1 = []
        cluster2 = []
        cluster3 = []
    
    # Teller voor unieke hut- en toilet-ID's
        hut_counter = 0
        toilet_counter = 0
        # extra counter voor de 5 badkamers! 
        badkamer_counter = 0 
    
    # Maak agenten voor elk compartimenttype en voeg ze toe aan de scheduler
        for i, compartiment_type in enumerate(compartiment_types):
            c = Compartiment(self.aantal_bemanningsleden + i, self, compartiment_type)
            self.schedule.add(c)        
     
            if compartiment_type in ["hut", "badkamer", "toilet"]:
                cluster1.append(c)
            elif compartiment_type in ["keuken", "kantine", "recreatiekamer"]:
                cluster2.append(c)
            elif compartiment_type in ["brug", "platform", "machinekamer"]:
                cluster3.append(c)

        print(f'cluster 1: {[c.compartiment_type for c in cluster1]}')
        print(f'cluster 2: {[c.compartiment_type for c in cluster2]}')
        print(f'cluster 3: {[c.compartiment_type for c in cluster3]}')

    
        # Plaats cluster 1: hutten en badkamers
        self.plaats_cluster(cluster1, 0, 0, hut_counter, toilet_counter, badkamer_counter)
        # # Plaats cluster 2: keuken, kantine en recreatiekamer
        self.plaats_cluster(cluster2, 6, 0, hut_counter, toilet_counter, badkamer_counter)
        # # Plaats cluster 3: brug, platform en machinekamer
        self.plaats_cluster(cluster3, 8, 0, hut_counter, toilet_counter, badkamer_counter)
    
    # Debugprint om compact_mapping te controleren
        print(f"final compact_mapping: {self.compact_mapping}")

    # Verplaats bemanningsleden naar hun startposities
        for agent in self.schedule.agents:
            if isinstance(agent, Bemanningslid):
                agent.functie_uitvoeren()

    def plaats_cluster(self, cluster, start_x, start_y, hut_counter, toilet_counter, badkamer_counter):
        x, y = start_x, start_y
        for compartiment in cluster:
            print(f'compartiment: {compartiment.compartiment_type} id: {compartiment.unique_id}')
            if y >= 10:
                x += 2
                y = 0

            ## dit onderste deel samenvoegen, eerst check if type is present multiple times, then take those and place 
            # def list_duplicates(cluster):

            if compartiment.compartiment_type == "hut":
                self.grid.place_agent(compartiment, (x, y))
                unique_hut_key = f"hut_{hut_counter}"
                self.compact_mapping[unique_hut_key] = (x, y)
                hut_counter += 1
            elif compartiment.compartiment_type == "toilet": 
                ### omdat de hutten en toiletten al naast elkaar zijn aangemaakt, is de y+=1 onderaan deze for loop genoeg om ze naast elkaar te plaatsen op de grid. 
                self.grid.place_agent(compartiment, (x, y)) 
                unique_toilet_key = f"toilet_{toilet_counter}"
                self.compact_mapping[unique_toilet_key] = (x, y)
                toilet_counter += 1
            ### hier denk ik nog een elif voor de badkamers, daar moeten er ook 5 voor zijn. Nu wordt er maar 1 geplaatst. 
            elif compartiment.compartiment_type == "badkamer":
                self.grid.place_agent(compartiment, (x,y))
                self.compact_mapping[f"badkamer_{badkamer_counter}"] = (x, y)
                badkamer_counter +=1 
            else: 
                self.grid.place_agent(compartiment, (x, y))
                self.compact_mapping[compartiment.compartiment_type] = (x, y)
            
            
            y += 1 


                      
            
    def get_keuken_bezetting(self):
        # Tel het aantal bemanningsleden in de keuken ter verificatie model
        keukens = [agent for agent in self.grid.get_all_cell_contents() if isinstance(agent, Bemanningslid)]
        return len(keukens) 
    
    def step(self):
        self.datacollecter.collect(self)
        self.schedule.step()

#%% de agent Bemanningslid
class Bemanningslid(Agent):
    "Een bemanningslid waarbij wordt bijgehouden of hij ziek is of niet"
    def __init__ (self,unique_id, model, shift, functie, initial_state = State.NIET_BESMET):
        super().__init__(unique_id, model)
        self.state = initial_state #bemmaningslid is in eerste instantie niet besmet
        self.hygiëne = random.uniform(0,1) #Waarde voor relatieve hygiëne van bemmaningslid
        self.immuniteit = random.uniform(0.7,1) #Waarde voor immuniteit/kans op ziek worden van bemanningslid (Bij norovirus relatief hoog?)
        self.functie = functie #Functie van het betreffende bemanningslid
        self.shift = shift
        self.rooster = self.rooster_toewijzen(shift,functie)  #Rooster die bij functie hoort
        #print(f"Bemanningslid {unique_id} aangemaakt met functie {functie} en shift {shift}. Rooster: {self.rooster}")
       
          
    
    def rooster_toewijzen(self,shift,functie):
        if functie == "platformmedewerker":
            if shift == 1: 
                return platformmedewerker_1
            elif shift == 2: 
                return platformmedewerker_2
            elif shift == 3: 
                return platformmedewerker_3

        elif functie == "brugmedewerker":
            if shift == 1: 
                return brugmedewerker_1
            elif shift == 2: 
                return brugmedewerker_2
            elif shift == 3: 
                return brugmedewerker_3

        elif functie == "machinemedewerker":
            if shift == 1: 
                return machinemedewerker_1
            elif shift == 2: 
                return machinemedewerker_2
            elif shift == 3: 
                return machinemedewerker_3

        elif functie == "schoonmaker":
            if shift == 1:
                return schoonmaker_1
            elif shift == 2:
                return schoonmaker_2
            elif shift == 3:
                return schoonmaker_3

        elif functie == "kok":
            return kok

  
        return []
    
        
   
    def functie_uitvoeren(self):
       
        #bepaal huidige uur in tijdscyclus van 24 uur
        huidige_uur = self.model.schedule.steps %24
        #bepaal huidige locatie agent volgens het werkrooster
        huidige_locatie = self.get_huidige_locatie(huidige_uur)
        
        print(f"Bemanningslid {self.unique_id} huidige locatie: {huidige_locatie}")
        print(f"Current compact_mapping: {self.model.compact_mapping}")
    
       
        if huidige_locatie:
            if huidige_locatie in self.model.compact_mapping:                
                huidige_positie = self.model.compact_mapping[huidige_locatie]
                self.model.grid.move_agent(self, huidige_positie)
            else: 
                print(f"Waarschuwing: locatie {huidige_locatie} heeft geen positie in compact_mapping")
            
    
    def get_huidige_locatie(self, huidige_uur):
            
        totale_uren = 0
        for locatie, duur in self.rooster:
            totale_uren += duur
            if huidige_uur < totale_uren:
                return locatie 
            
        return None
  
   
    def interactie(self):
        #deze functie zorgt voor dat bemanningsleden interactie met elkaar hebben en het compartiment waar ze zich in bevinden
        #bemanningsleden kunnen hierdoor virus krijgen als andere in hetzelfde compartiment besmet zijn
       
        #lijst van die agenten op die zelfde specifieke locatie:
        zelfde_compartiment = self.model.grid.get_cell_list_contents([self.pos])
        #het bereffende compartiment vinden
        compartiment = [agent for agent in zelfde_compartiment if isinstance(agent, Compartiment)]
       
        #Bepalen infectiepercentage van het compartiment
        compartiment_infectiepercentage = compartiment[0].infectiepercentage if compartiment else 0
       
        #for loop om door alle bemanningsleden op dezelfde locatie te lopen
        for andere in zelfde_compartiment:
            if andere != self and isinstance(andere,Bemanningslid):
                if self.state == State.NIET_BESMET:
                    infectiekans = (1- self.hygiëne) * (1 - self.immuniteit)
                   
                    if andere.state == State.BESMET_MET_VERSCHIJNSELEN or andere.state == State.BESMET_ZONDER_VERSCHIJNSELEN:
                        if random.random() < infectiekans:
                            self.state = State.BESMET_ZONDER_VERSCHIJNSELEN
                            compartiment[0].update_besmette_bemanningsleden(State.BESMET_ZONDER_VERSCHIJNSELEN)
                           
                    elif compartiment_infectiepercentage < infectiekans:
                        if random.random() < compartiment_infectiepercentage:
                            self.state = State.BESMET_ZONDER_VERSCHIJNSELEN
                            compartiment[0].update_besmette_bemanningsleden(State.BESMET_ZONDER_VERSCHIJNSELEN)
                               
    
              
              
    def step(self): 
        self.functie_uitvoeren()
        self.interactie()
#%% Compartiment klasse
class Compartiment(Agent):
    "De verschillende ruimtes gecodeerd als afzonderlijke agenten, omdat het"
    "per ruimte verschilt wat de interactie binnen ruimte is/"

    def __init__(self,unique_id,model, compartiment_type):
        super().__init__(unique_id,model)
        self.infectiepercentage = 0 #initieel infectiepercentage
        self.compartiment_type = compartiment_type
        self.besmette_bemanningsleden = 0 #initieel aantal besmette bemanningsleden
       
 
       
    def update_infectiepercentage(self):
        #het updaten van het infectiepercentage gebaseerd op het aantal besmette bemanningsleden die zich in het compartiment bevinden
        totaal_bemanningsleden = len([agent for agent in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(agent,Bemanningslid)])
        if totaal_bemanningsleden > 0:
            self.infectiepercentage = self.besmette_bemanningsleden / totaal_bemanningsleden
        else:
            self.infectiepercentage = 0
           
           
    def update_besmette_bemanningsleden(self,state):
        #als bemanningslid besmet is komt er 1 bij
        if state == State.BESMET_ZONDER_VERSCHIJNSELEN or state == State.BESMET_MET_VERSCHIJNSELEN:
            self.besmette_bemanningsleden += 1
        elif state == State.HERSTELD and self.besmette_bemanningsleden > 0:
            self.besmette_bemanningsleden -= 1

    def step(self):
        self.update_infectiepercentage()   
#%% Schip met compartimenten tekenen

def teken_compartimenten(grid): 
    pos = {}
    labels = {} 
    colors = [] 
    
    for cel in grid.G.nodes(): 
        compartimenten = grid.get_cell_list_contents([cel])
        x, y = cel 
        pos[(x, y)] = (y, -x)
        if compartimenten: 
            compartiment = compartimenten[0]
            labels[(x, y)] = compartiment.compartiment_type
            if compartiment.compartiment_type == "hut":
                colors.append("blue")
            elif compartiment.compartiment_type == "toilet": 
                colors.append("red")
            elif compartiment.compartiment_type == "badkamer":
                colors.append("green")
            elif compartiment.compartiment_type in ["keuken", "kantine", "recreatiekamer"]:
                colors.append("orange")
            elif compartiment.compartiment_type in ["brug", "platform", "machinekamer"]:
                colors.append("purple")
            else:
                colors.append("gray")
        else: 
            colors.append("white")
        
    plt.figure(figsize=(10, 10))
    nx.draw(grid.G, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=3000, font_size=8)
    plt.show()

#%%


def plot_keuken_bezetting(datacollector):
    data = datacollector.get_model_vars_dataframe()
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.xlabel("Tijd")
    plt.ylabel("Keuken Bezetting")
    plt.title("Keuken Bezetting Over Tijd")
    plt.show()
            
#%% Simulatie starten


virus_model = Virus(aantal_bemanningsleden=20)
for uur in range(48):
    virus_model.step()

plot_keuken_bezetting(virus_model.datacollector)
teken_compartimenten(virus_model.grid)
















