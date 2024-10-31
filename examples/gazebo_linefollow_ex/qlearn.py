import random
import pickle
import numpy as np


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        file = open(filename + ".pickle", 'rb')
        self.q = pickle.load(file)
        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        print("SAVING LINE\n")
        file = open(filename + ".pickle", 'wb')
        pickle.dump(self.q, file)

        # np.savetxt(filename + ".csv", self.q, delimiter=",")

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        chosenAction = self.actions[random.randint(0, len(self.actions) - 1)]

        if (random.random() >= self.epsilon):
            maxQ = self.q.get((state, self.actions[0]), 0.0)

            for action in self.actions[1:]:
                if (maxQ < self.q.get((state, action), 0.0)):
                    chosenAction = action

        if return_q:
            return chosenAction, self.q.get((state, chosenAction), 0.0)
        else:
            return chosenAction

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        maxQState2 = max(self.q.get(state2, action) for action in self.actions)
        currentQ = self.q.get((state1, action1), 0.0)

        bellman = self.alpha * (reward + self.gamma * maxQState2 - currentQ)

        if (state1, action1) in self.q:
            self.q[(state1, action1)] += bellman
        else:
            self.q[(state1, action1)] = bellman
