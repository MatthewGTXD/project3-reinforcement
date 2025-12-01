# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()


        for k in range(self.iterations):
            newValues = util.Counter()
            # to compute values[k+1] from values[k]
            # newValues represents the k+1 values, and will be computed in its entirety before replacing the old values
            for state in states:

                #handle cases without actions
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue

                actions = self.mdp.getPossibleActions(state)


                #find best q value for this state
                maxQValue = self.computeQValueFromValues(state, actions[0])
                for action in actions:
                    QValue = self.computeQValueFromValues(state, action)
                    if QValue > maxQValue:
                        maxQValue = QValue

                newValues[state] = maxQValue

            #overwrite old values after every state is calculated
            self.values = newValues








    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        accumulator = 0
        for transition in transitions:
            nextState = transition[0]
            probability = transition[1]
            #Q value formula by the book
            accumulator += ( (probability) * (self.mdp.getReward(state, action, nextState) + (self.discount * self.getValue(nextState)) ) )
        return accumulator

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)


        bestAction = actions[0]
        maxQValue = self.computeQValueFromValues(state, actions[0])
        for action in actions:
            QValue = self.computeQValueFromValues(state, action)
            if QValue > maxQValue:
                maxQValue = QValue
                bestAction = action

        return bestAction



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()


        for k in range(self.iterations):
            #each iteration now only changes 1 state value according to the index
            index = k % len(states)
            state = states[index]


            #handle cases without actions
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)


            #find best q value for this state
            maxQValue = self.computeQValueFromValues(state, actions[0])
            for action in actions:
                QValue = self.computeQValueFromValues(state, action)
                if QValue > maxQValue:
                    maxQValue = QValue

            self.values[state] = maxQValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        #compute predecessors
        states = self.mdp.getStates()

        predecessors = {}
        for s in states:
            predecessors[s] = []


        for s in states:
            if self.mdp.isTerminal(s):
                continue

            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                successors = self.mdp.getTransitionStatesAndProbs(s, a)
                for succ in successors:
                    #add s as the predecessor of the successor

                    if predecessors[succ[0]].count(s) == 0:
                        predecessors[succ[0]].append(s)

        pq = util.PriorityQueue()

        # maps states to their priority in the priority queue. It is set to 1 if they are not in the priority queue
        # This is helpful because later we have to do something based on the priority of certain states
        priorities = {}

        #push the states
        for s in states:

            actions = self.mdp.getPossibleActions(s)

            if len(actions) == 0:
                priorities[s] = 1
                continue

            # find best q value for this state
            maxQValue = self.computeQValueFromValues(s, actions[0])
            for action in actions:
                QValue = self.computeQValueFromValues(s, action)
                if QValue > maxQValue:
                    maxQValue = QValue

            diff = abs(maxQValue - self.values[s])

            pq.push(s, -diff)
            priorities[s] = -diff

        for i in range(self.iterations):
            # if the priority queue is empty, then terminate
            if pq.isEmpty():
                break

            #Pop s off of the queue
            s = pq.pop()
            priorities[s] = 1

            #If not a terminal state, update the value
            if not self.mdp.isTerminal(s):
                # find best q value for this state and update it
                actions = self.mdp.getPossibleActions(s)

                if len(actions) == 0:
                    continue

                maxQValue = self.computeQValueFromValues(s, actions[0])
                for action in actions:
                    QValue = self.computeQValueFromValues(s, action)
                    if QValue > maxQValue:
                        maxQValue = QValue

                self.values[s] = maxQValue

            #for each predecessor
            for p in predecessors[s]:
                #compute the difference as before, but for p
                actions = self.mdp.getPossibleActions(p)

                maxQValue = self.computeQValueFromValues(p, actions[0])
                for action in actions:
                    QValue = self.computeQValueFromValues(p, action)
                    if QValue > maxQValue:
                        maxQValue = QValue

                diff = abs(maxQValue - self.values[p])
                if diff > self.theta:
                    if -diff < priorities[p]:
                        pq.push(p, -diff)
                        priorities[p] = -diff






