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
        "*** YOUR CODE HERE ***"
        iter_count = 0
        while(iter_count < self.iterations):
            state_q = util.Counter()
            s_list = self.mdp.getStates()
            # each state s from all states:
            for s in s_list:
                if(not self.mdp.isTerminal(s)):
                    # compute highest q-value based on its possible actions
                    q_values = []
                    a_poss = self.mdp.getPossibleActions(s)
                    for a in a_poss:
                        q = self.computeQValueFromValues(s, a)
                        q_values.append(q)
                    max_q = max(q_values)
                    state_q[s] = max_q
            # update self.values
            self.values = state_q
            iter_count += 1





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
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        q = 0.0
        next_poss = self.mdp.getTransitionStatesAndProbs(state, action)
        # every next state with prob based on the action
        for pair in next_poss:
            next_s, prob = pair
            # add up
            q = q + prob*(self.mdp.getReward(state, action, next_s) + self.discount*self.getValue(next_s))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # No legal actions situation
        if(self.mdp.isTerminal(state)):
            return None
        # Normal situation
        q_values = []
        a_dict = {}
        a_poss = self.mdp.getPossibleActions(state)
        # every possible action
        for a in a_poss:
            q = self.computeQValueFromValues(state, a)
            q_values.append(q)
            a_dict[q] = a
        max_q = max(q_values)
        # select action that has highest q-value
        sel_a = a_dict[max_q]
        #print(sel_a)
        return sel_a





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
        "*** YOUR CODE HERE ***"
        iter_count = 0
        s_list = self.mdp.getStates()
        s_index = 0
        while (iter_count < self.iterations):
            # one state per iteration, cycle from all states
            if(s_index >= len(s_list)):
                s_index = 0
            s = s_list[s_index]
            # not terminal state:
            if (not self.mdp.isTerminal(s)):
                # compute highest q-value based on its possible actions
                q_values = []
                a_poss = self.mdp.getPossibleActions(s)
                for a in a_poss:
                    q = self.computeQValueFromValues(s, a)
                    q_values.append(q)
                max_q = max(q_values)
                # only update that state
                self.values[s] = max_q

            iter_count += 1
            s_index += 1

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
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        s_list = self.mdp.getStates()
        pre = {}
        for s in s_list:
            a_poss = self.mdp.getPossibleActions(s)
            for a in a_poss:
                reachable = self.mdp.getTransitionStatesAndProbs(s, a)
                for pair in reachable:
                    reach, prob = pair
                    if(prob != 0):
                        pre.setdefault(reach, []).append(s)

        # Initialize an empty priority queue.
        pq = util.PriorityQueue()

        # each nonterminal state s
        s_list = self.mdp.getStates()
        for s in s_list:
            if (not self.mdp.isTerminal(s)):
                q_values = []
                a_poss = self.mdp.getPossibleActions(s)
                for a in a_poss:
                    q = self.computeQValueFromValues(s, a)
                    q_values.append(q)
                max_q = max(q_values)
                diff = abs(self.values[s] - max_q)
                pq.push(s, diff*(-1))

        # each iteration
        iter_count = 0
        while (iter_count < self.iterations):
            if(pq.isEmpty()):
                return
            s = pq.pop()
            if(not self.mdp.isTerminal(s)):
                # compute highest Q-value
                q_values = []
                a_poss = self.mdp.getPossibleActions(s)
                for a in a_poss:
                    q = self.computeQValueFromValues(s, a)
                    q_values.append(q)
                max_q = max(q_values)

                self.values[s] = max_q
            # each predecessor p of s
            for p in pre[s]:
                # compute highest Q-value
                q_values = []
                a_poss = self.mdp.getPossibleActions(p)
                for a in a_poss:
                    q = self.computeQValueFromValues(p, a)
                    q_values.append(q)
                max_q = max(q_values)
                diff = abs(self.values[p] - max_q)
                if(diff > self.theta):
                    pq.update(p, diff*(-1))

            iter_count += 1


