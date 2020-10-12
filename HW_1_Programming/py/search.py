# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
      Returns the start state for the search problem
      """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure that you implement the graph search version of DFS,
    which avoids expanding any already visited states. 
    Otherwise your implementation may run infinitely!
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    """
    YOUR CODE HERE
    """
    # initialize closeset and stack
    closeSet = []
    repo = util.Stack()
    startState = problem.getStartState()
    repo.push((startState, []))
    # traverse the tree
    while not repo.isEmpty():
        (state, action) = repo.pop()
        if problem.isGoalState(state):
            return action
        # if find a state already in closeset, skip it
        if state not in closeSet:
            closeSet.append(state)
            # append all successors in stack
            for leaf in problem.getSuccessors(state):
                actions = list(action)
                actions.append(leaf[1])
                repo.push((leaf[0], actions))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """
    YOUR CODE HERE
    """
    # init queue and closeset
    closeSet = []
    repo = util.Queue()
    startState = problem.getStartState()
    repo.push((startState, []))
    # traverse the tree
    while not repo.isEmpty():
        (state, action) = repo.pop()
        # if the state already in closeset, skip it
        if problem.isGoalState(state):
            return action
        # enqueue all successors
        if state not in closeSet:
            closeSet.append(state)
            for leaf in problem.getSuccessors(state):
                actions = list(action)
                actions.append(leaf[1])
                repo.push((leaf[0], actions))
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """
    YOUR CODE HERE
    """
    # init closeset and a priorityqueue
    closeSet = []
    repo = util.PriorityQueue()
    startState = problem.getStartState()
    repo.push((startState, [], 0), 0)
    # traverse whole tree
    while not repo.isEmpty():
        (state, action, priority) = repo.pop()
        # if the state already in closeset, skip it
        if problem.isGoalState(state):
            return action
        # append all successors in priorityqueue
        if state not in closeSet:
            closeSet.append(state)
            for leaf in problem.getSuccessors(state):
                actions = list(action)
                actions.append(leaf[1])
                cost = priority + leaf[2]
                repo.push((leaf[0], actions, cost), cost)
                repo.update((leaf[0], actions, cost), cost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    YOUR CODE HERE
    """
    # initialize a priorotyqueue and find the heuristic distance from statestate to goal
    closeSet = []
    repo = util.PriorityQueue()
    startState = problem.getStartState()
    dist = heuristic(startState, problem)
    repo.push((startState, [], 0), dist)
    while not repo.isEmpty():
        (state, action, priority) = repo.pop()
        # if the state already in closeset, skip it
        if problem.isGoalState(state):
            return action
        if state not in closeSet:
            closeSet.append(state)
            # append all successors in pirorityqueu
            for leaf in problem.getSuccessors(state):
                actions = list(action)
                actions.append(leaf[1])
                cost = priority + leaf[2]
                hValue = heuristic(leaf[0], problem)
                repo.push((leaf[0], actions, cost), (cost + hValue))
                repo.update((leaf[0], actions, cost), (cost + hValue))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
