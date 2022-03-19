#particle swarm optimization for Schwefel minimization problem


#need some python libraries
import copy
import math
from random import Random


#to setup a random number generator, we will specify a "seed" value
seed = 51132021
myPRNG = Random(seed)
#to get a random number between 0 and 1, write call this:             myPRNG.random()
#to get a random number between lwrBnd and upprBnd, write call this:  myPRNG.uniform(lwrBnd,upprBnd)
#to get a random integer between lwrBnd and upprBnd, write call this: myPRNG.randint(lwrBnd,upprBnd)

w = 0.729844 # Inertia weight to prevent velocities becoming too large
c1 = 1.496180 # Scaling co-efficient on the social component
c2 = 1.496180 # Scaling co-efficient on the cognitive component
dimension = 2 # Size of the problem
iterations = 20000
swarmSize = 150
lowerBound = -500  #bounds for Schwefel Function search space
upperBound = 500   #bounds for Schwefel Function search space
Max_velocity = 500

      
#Schwefel function to evaluate a real-valued solution x    
# note: the feasible space is an n-dimensional hypercube centered at the origin with side length = 2 * 500
                       
def evaluate(x):          
    val = 0
    d = len(x)
    for i in range(d):
        val = val + x[i]*math.sin(math.sqrt(abs(x[i])))
                                        
    val = 418.9829*d - val         
                    
    return val          

def findGBest(pos):    
    gbest = pos[0]
    gbval = evaluate(pos[0])
    
    for i in range(0, len(pos)):
        cur = evaluate(pos[i])
        if cur < gbval:
            gbval = cur
            gbest = pos[i][:]
    return gbest

def updatePositions(pos, velocity):
    for i in range(dimension):
        pos[i] = pos[i] + velocity[i]
        if pos[i] > 500:
            pos[i] = 500 
        elif pos[i] < -500:
            pos[i] = -500     
    return pos
    
    
def findParticleVelocity(pos, pBest, prevVelocity, gbest):
        c1=1        # cognative constant
        c2=2        # social constant
        velocity = []
        for i in range(0,len(pos)):
            r1=myPRNG.random()
            r2=myPRNG.random()
 
            vel_cognitive=c1*r1*(pBest[i]-pos[i])
            vel_social=c2*r2*(gbest[i] - pos[i])
            velocityVal = w * prevVelocity[i] + vel_cognitive + vel_social
            if (velocityVal > Max_velocity):
                velocityVal = Max_velocity
            elif (velocityVal < - Max_velocity):
                velocityVal = - Max_velocity
            velocity.append(velocityVal)
            
        return velocity
          
   
#the swarm will be represented as a list of positions, velocities, values, pbest, and pbest values

pos = [[] for _ in range(swarmSize)]      #position of particles -- will be a list of lists; e.g., for a 2D problem with 3 particles: [[17,4],[-100,2],[87,-1.2]]
vel = [[] for _ in range(swarmSize)]      #velocity of particles -- will be a list of lists similar to the "pos" object 
#pBest = [[] for _ in range(swarmSize)] 
#note: pos[0] and vel[0] provides the position and velocity of particle 0; pos[1] and vel[1] provides the position and velocity of particle 1; and so on. 


curValue = [] #evaluation value of current position  -- will be a list of real values; curValue[0] provides the evaluation of particle 0 in it's current position
pbest = []    #particles' best historical position -- will be a list of lists: pbest[0] provides the position of particle 0's best historical position
pbestVal = [] #value of pbest position  -- will be a list of real values: pbestBal[0] provides the value of particle 0's pbest location
gBest =[]
gBestVal = []
var_gBestVal = 10000

#initialize the swarm randomly
for i in range(swarmSize):
    for j in range(dimension):
        pos[i].append(myPRNG.uniform(lowerBound,upperBound))    #assign random value between lower and upper bounds
        vel[i].append(myPRNG.uniform(-1,1))                     #assign random value between -1 and 1   --- maybe these are good bounds?  maybe not...
    curValue.append(evaluate(pos[i]))   #evaluate the current position
                                                    
pbest = pos[:]          # initialize pbest to the starting position
pbestVal = curValue[:]  # initialize pbest to the starting position

for j in range(0, iterations):#Note indexes switched convention
    
    gBest = findGBest(pos)
    for i in range(swarmSize):
        vel[i] = findParticleVelocity(pos[i], pbest[i], vel[i], gBest)
        pos[i] = updatePositions(pos[i], vel[i])
        
    for i in range(swarmSize):
        tempPBest = evaluate(pos[i])
        if tempPBest < pbestVal[i]:
            pbest[i] = pos[i][:]
            pbestVal[i] = tempPBest
    
    
    if var_gBestVal > evaluate(gBest):
        print ("Current Iteration:", j)
        print("Best value obtained:",evaluate(gBest))
        print("Solution of best value obtained", gBest)
        var_gBestVal = evaluate(gBest)
        
