import copy
import math
from random import Random
import numpy as np
import time
start = time.process_time()


class GeneticAlgorithm():
    def __init__(
        self, 
        seed, 
        lowerBound, 
        upperBound,
        dimensions, 
        populationSize,
        generationSize,
        crossOverRate,
        mutationRate,
        elitismRate
        ) -> None:
        """
        Instantiates class variables with passed argument values

        Args:
            seed (int): random seed value to reproduce results
            lowerBound (int): lower bound for search space
            upperBound (int): upper bound for search space
            dimensions (int): dimensionality of search space
            populationSize (int): size of population
            generationSize (int): number of generations
            crossOverRate (float): cross over rate
            mutationRate (float): mutation rate
            elitismRate (float): rate to select elite parent genes
        """
        self.myPRNG = Random(seed)
        self.lowerBound         = lowerBound
        self.upperBound         = upperBound
        self.dimensions         = dimensions
        self.populationSize     = populationSize
        self.generationSize     = generationSize
        self.crossOverRate      = crossOverRate
        self.mutationRate       =  mutationRate
        self.elitismRate        =  mutationRate
                
        
    def createChromosome(self) -> None:
        return [self.myPRNG.uniform(self.lowerBound, self.upperBound) for _ in range(self.dimensions)]
        
    def initializePopulation(self):
        population = []
        for i in range(self.populationSize):
            chromosome = self.createChromosome()
            population.append((chromosome, self.evaluate(chromosome)))
        
        population = sorted(population, key= lambda population: population[1])
        
        return population
        
    def crossover(self, x1, x2) :
        """generates 2 offsprings from x1, x2

        Args:
            x1 (list): parent 1
            x2 (list): parent 2

        Returns:
            tuple: returns tuple of 2 offsprings from x1, x2
        """
        crossOverPt = self.myPRNG.randint(1, len(x1)-1)
        rand_num    = self.myPRNG.uniform(0, 1)
        if rand_num < self.crossOverRate:
            beta = self.myPRNG.random()
        else:
            beta = 0.5
        
        cross_over_one = (np.array(x1)-beta*(np.array(x1)-np.array(x2))).tolist()
        cross_over_two = (np.array(x1)+beta*(np.array(x1)-np.array(x2))).tolist()
        
        d = len(x1)
        if crossOverPt < d/2:
            offspring1 = x1[0:crossOverPt] + cross_over_one[crossOverPt:d]
            offspring2 = x2[0:crossOverPt] + cross_over_two[crossOverPt:d]

        else:
            offspring1 = cross_over_one[0:crossOverPt] + x1[crossOverPt:d]
            offspring2 = cross_over_two[0:crossOverPt] + x2[crossOverPt:d]

        return offspring1, offspring2

    def evaluate(self, chromosome):
        return 418.9829*len(chromosome)  - sum([chromosome[i]*np.sin(np.sqrt(np.abs(chromosome[i]))) for i in range(len(chromosome))])

    def tournamentSelection(self, population, k):

        matingPool = []
    
        while len(matingPool)<self.populationSize:
            
            ids = [self.myPRNG.randint(0,self.populationSize-1) for i in range(k)]
            competingIndividuals = [population[i][1] for i in ids]
            bestID=ids[competingIndividuals.index(min(competingIndividuals))]
            matingPool.append(population[bestID][0])

        return matingPool

    def mutate(self, chromosome):

        if self.myPRNG.random() < self.mutationRate:
            temp = self.myPRNG.randint(0, len(chromosome)-1)
            chromosome[temp] = self.myPRNG.uniform(-500, 500)

        return chromosome

    def breeding(self,matingPool):

        children = []
        childrenFitness = []
        for i in range(0,self.populationSize-1,2):
            child1,child2=self.crossover(matingPool[i],matingPool[i+1])
            
            child1=self.mutate(child1)
            child2=self.mutate(child2)
            
            children.append(child1)
            children.append(child2)
            
            childrenFitness.append(self.evaluate(child1))
            childrenFitness.append(self.evaluate(child2))
            
        tempZip = zip(children, childrenFitness)
        popVals = sorted(tempZip, key=lambda tempZip: tempZip[1])
        return popVals

    def insert(self, population, kids):

        elite_len = int(self.elitismRate*self.populationSize)
        elite_sols = population[0:elite_len+1]
        elite_sols +=kids[0:self.populationSize-elite_len+1]
        elite_sols = sorted(elite_sols,key = lambda elite_sols : elite_sols[1])
        return elite_sols

    def summaryFitness(self, pop):
        a=np.array(list(zip(*pop))[1])
        return np.min(a), np.mean(a), np.var(a)


ga = GeneticAlgorithm (
                seed=51132021,
                lowerBound = -500, 
                upperBound = 500,
                dimensions = 200, 
                populationSize = 60,
                generationSize = 200,
                crossOverRate= 0.8,
                mutationRate = 0.2,
                elitismRate = 0.4
                    )


Population = ga.initializePopulation()

for j in range(ga.generationSize):
    mates=ga.tournamentSelection(Population,3)
    Offspring = ga.breeding(mates)
    Population = ga.insert(Population, Offspring)
    minVal,meanVal,varVal=ga.summaryFitness(Population)  #check out the population at each generation

print (ga.summaryFitness(Population))
print(Population[0])
print("\n",time.process_time() - start)