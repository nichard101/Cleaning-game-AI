__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np
import random
import csv

fitness_array = []
training_sessions = 50
session = 1
mutation_chance = 50
debug = True

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", training_sessions), ("self", 1)]    # Train against random agent for 5 generations,
                                                            # then against self for 1 generation


# This is the class for your cleaner/agent
class Cleaner:

    chromosome = []
    bias = []
    sum_energy = 0


    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)

        # Leave these variables as they are, even if you don't use them in your AgentFunction - they are
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns
        self.chromosome = []
        self.bias = []
        self.sum_energy = 0

        self.generateChromosome()
        #print("{} {}".format(len(self.chromosome), len(self.chromosome[0])))

    def generateChromosome(self):
        for i in range(4):
            temp = []
            for j in range(63):
                temp.append(random.randint(-10,10))
            self.bias.append(random.randint(-50,50))
            self.chromosome.append(temp)

    def AgentFunction(self, percepts):

        # The percepts are a tuple consisting of four pieces of information
        #
        # visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
        #          is a 3x5x4 tensor, giving four maps with different information
        #          - the dirty,clean squares
        #          - the energy
        #          - the friendly and enemy cleaners that are able to traverse vertically
        #          - the friendly and enemy cleaners that are able to traverse horizontally
        #
        #  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
        #           the cleaner can still perform before it runs out of charge
        #
        #  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        #
        #  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
        #          backward movement might fail if it would result in a collision with another robot); fails=0 means
        #          the last action succeeded.


        visual, energy, bin, fails = percepts

        self.sum_energy += energy

        visual_flattened = [item for sublist in visual for sub in sublist for item in sub]

        visual_flattened.append(energy)
        visual_flattened.append(bin)
        visual_flattened.append(fails)
        # You can further break down the visual information

        floor_state = visual[:,:,0]   # 3x5 map where -1 indicates dirty square, 0 clean one
        energy_locations = visual[:,:,1] #3x5 map where 1 indicates the location of energy station, 0 otherwise
        vertical_bots = visual[:,:,2] # 3x5 map of bots that can in this turn move up or down (from this bot's point of
                                      # view), -1 if the bot is an enemy, 1 if it is friendly
        horizontal_bots = visual[:,:,3] # 3x5 map of bots that can in this turn move up or down (from this bot's point
                                        # of view), -1 if the bot is an enemy, 1 if it is friendly

        #You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        # a mape where -1 indicates dirty square, 0 a clean one, and 1 an energy station.

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

        # The index of the largest value in the 'actions' vector/list is the action to be taken,
        # with the following interpretation:
        # largest value at index 0 - move forward;
        # largest value at index 1 - turn right;
        # largest value at index 2 - turn left;
        # largest value at index 3 - move backwards;
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # .
        # .
        # .

        # Right now this agent ignores percepts and chooses a random action.  Your agents should not
        # perform random actions - your agents' actions should be deterministic from
        # computation based on self.chromosome and percepts



        # action_vector = np.random.randint(low=-100, high=100, size=self.nActions)

        #print(len(visual_flattened))
        action_vector = np.add(Multiply(self.chromosome,visual_flattened), self.bias)
        #print(len(action_vector))
        return action_vector

def Multiply(matrix, vector):
    out = []
    for v in matrix:
        out.append(DotProduct(v,vector))
    return out

def DotProduct(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = []

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, c in enumerate(population):
        # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
        # game engine, which is a dictionary with the following information on the performance of the cleaner in
        # the last game:
        #
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        # This fitness functions considers total number of cleaned squares.  This may NOT be the best fitness function.
        # You SHOULD consider augmenting it with information from other stats as well.  You DON'T HAVE TO make use
        # of every stat.

        number_of_rotations = c.game_stats['successful_actions'] - c.game_stats['visits']
        energy_per_recharge = 0
        if c.game_stats['recharge_count']!=0:
            energy_per_recharge = float(c.game_stats['recharge_energy']) / float(c.game_stats['recharge_count'])
        total_not_emptied = c.game_stats['cleaned'] - c.game_stats['emptied']
        average_energy = float(c.sum_energy) / float(c.game_stats['active_turns'])
        ratio_cleaned = float(c.game_stats['cleaned']) / float(c.game_stats['visits'])

        cleanedWeight = 100
        emptiedWeight = 80
        ratioCleanedWeight = 0
        notEmptiedWeight = 30
        rotationsWeight = 5
        activeTurnsWeight = 40
        energyRatioWeight = 0
        energySumWeight = 0
        energyAverageWeight = 0

        fitness.append(c.game_stats['cleaned']*cleanedWeight) # + c.game_stats['emptied']*emptiedWeight + total_not_emptied*notEmptiedWeight + c.game_stats['active_turns']*activeTurnsWeight + number_of_rotations*rotationsWeight)

    return fitness

def chooseParent(old_population):
    fitness = evalFitness(old_population)
    sumFitness = sum(fitness)
    randNum = random.uniform(0,sumFitness)
    cumulative = 0
    for agent, weight in zip(old_population,fitness):
        cumulative += weight
        if cumulative >= randNum:
            return agent


def newGeneration(old_population):

    # This function should return a tuple consisting of:
    # - a list of the new_population of cleaners that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    # Fetch the game parameters stored in each agent (we will need them to
    # create a new child agent)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns
    
    fitness = evalFitness(old_population)

    unsortedList = [(fitness[i], old_population[i]) for i in range(len(fitness))]
    unsortedList.sort(key=lambda x:x[0])

    old_pop_2 = [agent[1] for agent in unsortedList[-int(0.5*len(unsortedList)):]]

    # At this point you should sort the old_population cleaners according to fitness, setting it up for parent
    # selection.
    # .
    # .
    # .

    # Create new population list...
    new_population = [agent[1] for agent in unsortedList[-int(0.1*len(unsortedList)):]]
    for n in range(len(old_population)-len(new_population)):

        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        parent1 = chooseParent(old_pop_2)
        parent2 = chooseParent(old_pop_2)

        c1 = parent1.chromosome
        c2 = parent2.chromosome

        newChromosome = []

        for i in range(4):
            temp = []
            while len(temp) < 63:
                n = np.random.randint(0,63)
                j = 0
                while j < n:
                    temp.append(c1[i][j])
                    if np.random.random()*100 <= mutation_chance:
                        temp[-1] = np.random.randint(-10,10)
                    j += 1
                while j < 63:
                    temp.append(c2[i][j])
                    if np.random.random()*100 <= mutation_chance:
                        temp[-1] = np.random.randint(-10,10)
                    j += 1
            
            newChromosome.append(temp)
        
        new_cleaner.chromosome = newChromosome

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.

        # .
        # .
        # .

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    fitness_array.append(avg_fitness)
    if training_sessions == sessions and debug == True:
        with open("{}%mutation".format(mutation_chance), mode='w',newLine='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(fitness_array)
    else:
        sessions = sessions + 1

    return (new_population, avg_fitness)
