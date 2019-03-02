# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 19:01:01 2018

@author: Praveen
"""
# i had to create an uncommented ai.py file because this file has a lot of comment and thus im having some trouble running map through this one.
# the code works fine but the python interpreter finds it hard to diffentiate between modules and varaibles.

#import libraries.
import numpy as np #for arrays
import random #for random points on the enviornment
import os # using system methods for our program
import torch #deep learning lib
import torch.nn as nn#returns Q values for actions
import torch.nn.functional as F #functional modules from torch... includes deep learning functions
import torch.optim as optim #optimizer for gradient descent
import torch.autograd as autograd # used for advanced arrays which are used in gradients and convergence.
from torch.autograd import Variable #we need a variable after autograd which contains gradients...so we need its own variable.


#creating architecture of Neural network
#for more details, read this documentation from pytorch : https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html
class network(nn.Module): #inherit all the modules from nn class (as a parent class)
      #init will have input neurons and output neurons
      def __init__(self, input_size, nb_action): #self is used before to identity the object. when we hve more objects, its hard to define variables for these many objects so we use self method
            super(network,self).__init__() # using super class modules
            #specifying the input layer
            self.input_size = input_size # sizde of input layer of the network: 5 neurons
            self.nb_action = nb_action # size of output layer of the network : 3 neurons
            #defining the hidden layer 
            #connectoon between layers (2 full connections...input to hidden and hidden to output)
            self.fc1 = nn.Linear(input_size, 30) #first full connection
            self.fc2 = nn.Linear(30, nb_action) #second full connection.
      
      def forward_propagation(self,state):# the state is basically the input....but when we move forward...we call it a state rather than input .
            #returns the nb-action...basically Q values for each state that our AI takes.
            #it returns actions based on state.
            x = F.relu(self.fc1(state)) #this is the rectifier function.(x variable basically represents the current neuron....so we can use it to activate neuron.) . reLU produces a stright line from -x to 0 and a shape slop from 0 to +xy axis
            q_values = self.fc2(x) #calculating the output value from state (getting q avlues from x to fc2)
            return q_values


#implement replay (based on markov decision process)
# we put 100 points into the memory for the code to store which is much better than storing just one
#then we randomly select a subset(sample) of these points and then use it for next move
class replay_memory(object):
      def __init__(self,capacity):
            self.capacity = capacity #assignning capacity
            self.memory = [] # giving memory to these capacity elements
              
      def push(self,event): #creates an event which we will append in the memory
            #event is a 4 touple.... i hv defined it in later part of the code.
            #we need a list to append something
            self.memory.append(event)
            #if we have more events than expected....we use a condition
            if len(self.memory) > self.capacity:
                  del self.memory[0] #if more than capacity then we delete the first element 
      
      def random_sample(self,batch_size):#we want a batch(subset) sample from the 100 points
            samples = zip(*random.sample(self.memory, batch_size)) #contains the sample of the memory ...size of sample is the size of our memory
            #Example of zip function : if LIST_A= ((1,2,3),(4,5,6))...then after applying zip function...the OUTPUT_LIST = ((1,4),(2,5),(3,6))
            #events have the form : state,action and the reward...but for algorithm we dont want this format
            #we want different samples for different operations....ie one batch sample for states, one batch sampele for actions and one batch sample for rewards
            #example : consider two lists : LIST_A = ((state1,action1,reward1),(state2,action2,reward2))
            #we want the output to be in batches : OUTPUT_LIST=((state1,state2),(action1,action2),(reward1,reward2))
            
            #we then wrap these into a pytorch variable. a variable that consists of both (torch,gradient)
            # we put each of these new batches into pytorch variable and then assign a gradient to each batch
            return map(lambda x: Variable(torch.cat(x, 0)), samples) #fuction maps to the varaible. #first dimension has value 0
            #for each batch which is containted in x,we have to concatinate them in such a way that they lie in one dimension and now 2 dimension as they are now
                       

#implementing DEEP Q LEARNING

class brainOfCar():
      def __init__(self,input_size,nb_action,gamma):
            self.gamma = gamma
            self.reward_window = [] # our memory is tracked by this window....it shift as the car moves from one state to another
            self.model = network(input_size,nb_action) #initializing variable of the network class.
            self.memory = replay_memory(100000) #capacity=100000, this corresponds to number of transitions(number of events) : last state,new state,last action and last reward 
            self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #even rmsoptimizer can be choosen from the the functiions
            #another parameter in Adam is lr (Leanring rate)
            self.last_state = torch.Tensor(input_size).unsqueeze(0) #we hv 5 dimensions : straight,rightmleft,oriantation,-orientation....we create single dimension using unsqueeze function. arg 0 in unsqueeze means state dimensions
            #since last state will be the new input in the network...it can not simply be a vector...it shoul be in a batch. the network only accepts a batch of observations
            self.last_action =  0#based on the rotation
            self.last_reward = 0
      
      #now we need a select function which will help us to select the right move 
      def select_action(self,state): #action is based on the state
            #we need to select the best function but at the same time we need to explore others also becuase maybe they might lead us to a better, efficient and productive goal state
            #to do exactly this... ill use soft max function.
            #we use Q value function...(state,action)
            #we have 3 Q values...(left,straight and right)
            #we are generating probabilties of each of them and then see which is the best one
            probs = F.softmax(self.model(Variable(state, volatile = True))*150) # parameters are those for which we want to generate the probabilities. so we need to use model because it contains input networks            
            # we use volatile because model object creates an network which is of tensor variable type...so we need to  convert back to normal variable.
            # we also used temperature parameter which basically helps in selection of action.
            #example : softmax((1,2,3)) = [0.01,0.11,0.85] => softmax((1,2,3)*3) = [0,0.02.0.98]
            action = probs.multinomial() #gives random draw from probs
            return action.data[0,0] #returning the data present in 1 dimension ....we unsqueezed it before so we need to do this.
      
      #learn function
      def learn(self,batch_state,batch_next_state,batch_reward,batch_action): #MDP(markov decision process)
            outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #self.model will return the whole batch of set of inputs...that is ...all the input states ... but we want only the best one..so we choose only one and then do the action...so we r using gather function
            #arg 1 in unsqueeze means action dimension , and then we kill the whole dimension n convert it into simple vector...so we use squeeze function
            next_outputs = self.model(batch_next_state).detach().max(1)[0] #check handbook for this.... based on the next output state Q(a,s subscript(n+1))
            #the nextoutput line is to be understood as : we select the max value from the state dimension and return the action associated with it which is present at dimension 1
            target = self.gamma*next_outputs + batch_reward #target= reward + gamma*nextoutput (given in handbook)           
            td_loss = F.smooth_l1_loss(outputs, target)  #temporal difference....best loos function in Deep q learning
            #now we to backpropagAte this loss to update the weights
            self.optimizer.zero_grad()
            #we reinitial the optimizer at every iteration because it updates the weight using back propagation everytimie .
            td_loss.backward(retain_variables = True) #use of retain variable is to freeze some memory while back propagating
            self.optimizer.step() #step backpropagates and optimizer updates the weight
      
      
      #updation ..state,action and rewards
      def update(self,reward,new_signal): #important function bcq it makes connection betwwen the game n the AI
            #this function is called in the GAME class of the map.py file. so this should update the values based on the last action and last signal
            new_state = torch.Tensor(new_signal).float().unsqueeze(0) #new state depends upon the signal that our car send....less sand or no sand or more sand
            #signal in the GAME class has 5 args which is a simple list. so we convert it into tensor variable and use it.
            #next we have to update the memory based on the transition(s(t),s(t+1),r(t),a(t))...this will give us new transition...we dont hv s(t+1)
            self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #last action [0,1,2]...so we convert into a torch tensor of long type which is used to read integer type                        
            action = self.select_action(new_state) # select action based on the state
            #if we hv 100 elemts in memory...we need to learn then...
            if len(self.memory.memory) > 100 : #memory object of replay memory function object
                  # before we learn, 
                  batch_state,batch_next_state,batch_action,batch_reward = self.memory.sample(100) # sample 100 uits of each category
                  self.learn(batch_state,batch_next_state,batch_reward,batch_action) # we learn every 100 samples of new states,rewards and actions
            #now after learning, we need to update last action 
            self.last_action = action
            #we reached a new state by using new_state, but we have not updated the last_state variable.
            self.last_state = new_state
            #now updating reward
            self.last_reward = reward
            # now updating the reward window (sliding window of 100 obs)
            self.reward_window.append(reward)
            if len(self.reward_window) > 1000: # to make sure reward_window never gets more than 1000 samples of rewards
                  del self.reward_window[0]
            # return the final action taken after updating
            return action
      
      #score function
      def score(self):
            #score = sumof(score in reward window) / total no. of elements in the winodw
            return sum(self.reward_window)/(len(self.reward_window)+1.) # to make sure denominator is never zero..we add 1
      
      def save(self): # to save them model and optimizer....bcq we want to save the weights so that we can come back at them anytime
            # ill use python dictonary
            torch.save({'state_dict' : self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        },'last_brain.pth') # saving to file "last_brain"
          
      #loading the saved model
      def load(self):
            #checking if file is present or not
            if os.path.isfile('last_brain.pth'):
                  print("=> loading checkpoint...") #hint for the user 
                  checkpoint = torch.load('last_brain.pth') #loading the file.
                  #now we have to update the model wtih the weights present in the file
                  #load_state_dict helps us to do this
                  self.model.load_state_dict(checkpoint['state_dict']) # we stored the state as dictory in save function
                  self.optimizer.load_state_dict(checkpoint['optimizer']) # stored the optimzer in the dictory of save function
                  print("Done !")
            else:
                  print("No checkpoint found ! ")
            
            