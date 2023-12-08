
import math

def aStar(worldView, startX, startY, goalX, goalY, filterFunction):
	if startX == goalX and startY == goalY:
		return None
	# Array of nodes that each consist of [x, y, pathLength, heuristicCost]
	# Initially contains only the start
	openList = [[startX, startY, 0, 0]]
	# Create an array with the same width and height as worldView and fill it with False
	# Is used to check which tiles are already searched
	walked = [[False for _ in range(len(worldView[0]))] for _ in range(len(worldView))]
	# Create an array with the same width and height as worldView and 
	parents = [[None for _ in range(len(worldView[0]))] for _ in range(len(worldView))]
	while len(openList) > 0:
		exp = openList.pop(0)
		x = exp[0]
		y = exp[1]
		walked[x][y] = True
		pathLength = exp[2]
		
		if x == goalX and y == goalY:
			path = [[goalX, goalY]]
			parent = parents[x][y]
			while not(parent[0] == startX and parent[1] == startY):
				#path.append(parent)
				path.insert(0, parent)
				parent = parents[parent[0]][parent[1]]
			return path
		offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]]
		
		for of in offsets:
			nx = x+of[0]
			ny = y+of[1]
			# If the new coordinate is in the boundary
			if -1 < nx < len(worldView) and -1 < ny < len(worldView[0]):
				if not(walked[nx][ny]) and ((goalX == nx and goalY == ny) or filterFunction(worldView[nx][ny], nx, ny)) and not(contains(nx, ny, openList)):
				# if not(walked[nx][ny]) and ((goalX == nx and goalY == ny) or not(filterFunction(worldView[nx][ny], nx, ny))) and not(contains(nx, ny, openList)):
					heurisicCost = math.fabs(x-goalX)+math.fabs(y-goalY)
					# Add the node to the list (ordered)
					index = 0
					for node in openList:
						if not(heurisicCost+pathLength > node[2]+node[3]):
							break
						index = index+1
					openList.insert(index, [nx, ny, pathLength+1, heurisicCost])
					parents[nx][ny] = [x, y]
	
	return None

def contains(x, y, openList):
	for node in openList:
		if node[0] == x and node[1] == y:
			return True
	return False
