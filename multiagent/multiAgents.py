# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        # Make pacman keep moving
        if (newPos == currentGameState.getPacmanPosition()):
            score = score - 100

        distanceGhost = []
        timeScared = []
        for i in range(len(newGhostStates)):
            if(not newScaredTimes[i]):
                g = newGhostStates[i]
                distanceGhost.append(manhattanDistance(newPos, g.getPosition()))
            else:
                timeScared.append(newScaredTimes[i])
        if(distanceGhost):
            # For closest ghost distance, longer distance better
            score = score + min(distanceGhost)
        if(timeScared):
            # For scared time, longer time better
            score = score + max(timeScared)

        distanceFood = []
        for foodPos in newFood.asList():
            distanceFood.append(manhattanDistance(newPos, foodPos))
        if (distanceFood):
            # For closest food, less distance better
            score = score - min(distanceFood)

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth):
            # terminal state
            if(gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            # max state: agent pacman, 0
            if(agentIndex == 0):
                return max_value(gameState, depth)
            # min state: agent ghost, >=1
            if(agentIndex >= 1):
                return min_value(gameState, agentIndex, depth)

        def min_value(gameState, agentIndex, depth):
            v = float('inf')
            # Successors
            for action in gameState.getLegalActions(agentIndex):
                # not reach all agents: not reach all agent ghost
                if(agentIndex+1 < gameState.getNumAgents()):
                    nextIndex = agentIndex + 1
                    v = min(v, value(gameState.generateSuccessor(agentIndex, action), nextIndex, depth))
                # all agent ghost reached, now to agent player-pacman
                else:
                    # update depth
                    updateDepth = depth + 1
                    v = min(v, value(gameState.generateSuccessor(agentIndex, action), 0, updateDepth))
            return v

        def max_value(gameState, depth):
            v = float('-inf')
            # Successors
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action), 1, depth))
            return v

        max_ = float('-inf')
        candidate_act = {}
        for action in gameState.getLegalActions(0):
            v = value(gameState.generateSuccessor(0, action), 1, 0)
            if(v > max_):
                candidate_act[v] = action
                max_ = v

        minimax_act = candidate_act[max_]
        return minimax_act
        #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, depth, alpha, beta):
            # terminal state
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            # max state: agent pacman, 0
            if (agentIndex == 0):
                return max_value(gameState, depth, alpha, beta)
            # min state: agent ghost, >=1
            if (agentIndex >= 1):
                return min_value(gameState, agentIndex, depth, alpha, beta)

        def min_value(gameState, agentIndex, depth, alpha, beta):
            v = float('inf')
            # Successors
            for action in gameState.getLegalActions(agentIndex):
                # not reach all agents: not reach all agent ghost
                if (agentIndex + 1 < gameState.getNumAgents()):
                    nextIndex = agentIndex + 1
                    v = min(v, value(gameState.generateSuccessor(agentIndex, action),
                                     nextIndex, depth, alpha, beta))
                # all agent ghost reached, now to agent player-pacman
                else:
                    # update depth
                    updateDepth = depth + 1
                    v = min(v, value(gameState.generateSuccessor(agentIndex, action),
                                     0, updateDepth, alpha, beta))
                if(v < alpha):
                    return v
                beta = min(v,beta)
            return v

        def max_value(gameState, depth, alpha, beta):
            v = float('-inf')
            # Successors
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action),
                                 1, depth, alpha, beta))
                if(v > beta):
                    return v
                alpha = max(v, alpha)
            return v

        max_ = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        candidate_act = {}
        for action in gameState.getLegalActions(0):
            v = value(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if (v > max_):
                candidate_act[v] = action
                max_ = v
            alpha = max_

        alphabeta_act = candidate_act[max_]
        return alphabeta_act


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, depth):
            # terminal state
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            # max state: agent pacman, 0
            if (agentIndex == 0):
                return max_value(gameState, depth)
            # min state: agent ghost, >=1
            if (agentIndex >= 1):
                return exp_value(gameState, agentIndex, depth)

        def exp_value(gameState, agentIndex, depth):
            v = 0
            # Successors
            for action in gameState.getLegalActions(agentIndex):
                p = 1 / len(gameState.getLegalActions(agentIndex))
                # not reach all agents: not reach all agent ghost
                if (agentIndex + 1 < gameState.getNumAgents()):
                    nextIndex = agentIndex + 1
                    v += p * value(gameState.generateSuccessor(agentIndex, action), nextIndex, depth)
                # all agent ghost reached, now to agent player-pacman
                else:
                    # update depth
                    updateDepth = depth + 1
                    v += p * value(gameState.generateSuccessor(agentIndex, action), 0, updateDepth)
            return v

        def max_value(gameState, depth):
            v = float('-inf')
            # Successors
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action), 1, depth))
            return v

        max_ = float('-inf')
        candidate_act = {}
        for action in gameState.getLegalActions(0):
            v = value(gameState.generateSuccessor(0, action), 1, 0)
            if (v > max_):
                candidate_act[v] = action
                max_ = v

        expectimax_act = candidate_act[max_]
        return expectimax_act

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()

    distanceGhost = []
    timeScared = []
    for i in range(len(ghostStates)):
        if (not scaredTimes[i]):
            g = ghostStates[i]
            distanceGhost.append(manhattanDistance(pos, g.getPosition()))
        else:
            timeScared.append(scaredTimes[i])
    if (distanceGhost):
        # For closest ghost distance, longer distance better
        # Here I lower the importance of closest ghost distance compared to closest food
        # Then the pacman can make more efficient action for eating food
        score = score + min(distanceGhost)*0.9
    if (timeScared):
        # For scared time, longer time better
        score = score + max(timeScared)

    distanceFood = []
    for foodPos in food.asList():
        distanceFood.append(manhattanDistance(pos, foodPos))
    if (distanceFood):
        # For closest food, less distance better
        # Here I strengthen the importance of closest food distance compared to closest ghost
        # Then the pacman can make more efficient action for eating food
        score = score - min(distanceFood)*1.1

    return score

# Abbreviation
better = betterEvaluationFunction
