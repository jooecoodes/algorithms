# [SORTING ALGORITHMS]



# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Example usage
# arr = [64, 34, 25, 12, 22, 11, 90]
# print(bubble_sort(arr))



#--------------------------------------------------------------------#


# Merge Sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr

# Example usage
# arr = [64, 34, 25, 12, 22, 11, 90]
# print(merge_sort(arr))

#--------------------------------------------------------------------#



# Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
# arr = [64, 34, 25, 12, 22, 11, 90]
# print(quick_sort(arr))



#---------------------------------------------------------------------#
#[SEARCHING ALGORITHMS]


# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example usage
# arr = [2, 3, 4, 10, 40]
# target = 10
# print(binary_search(arr, target))


# [DYNAMIC PROGRAMMING]


# Climbing Stairs 
def climb_stairs(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# Example usage
# print(climb_stairs(5))

# Longest Increasing Subsequence
def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# Example usage
# nums = [10, 9, 2, 5, 3, 7, 101, 18]
# print(length_of_lis(nums))

# [GREEDY ALGORITHMS]

# Jump Game

def can_jump(nums):
    max_reachable = 0
    for i in range(len(nums)):
        if i > max_reachable:
            return False
        max_reachable = max(max_reachable, i + nums[i])
    return True

# Example usage
# nums = [2, 3, 1, 1, 4]
# print(can_jump(nums))

# Gas Station

def can_complete_circuit(gas, cost):
    total_gas, total_cost, current_gas, start_index = 0, 0, 0, 0
    for i in range(len(gas)):
        total_gas += gas[i]
        total_cost += cost[i]
        current_gas += gas[i] - cost[i]
        if current_gas < 0:
            start_index = i + 1
            current_gas = 0
    return start_index if total_gas >= total_cost else -1

# Example usage
# gas = [1, 2, 3, 4, 5]
# cost = [3, 4, 5, 1, 2]
# print(can_complete_circuit(gas, cost))

# [GRAPH ALGORITHMS]

# Depth First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# Example usage
# graph = {
#     0: [1, 2],
#     1: [0, 3, 4],
#     2: [0, 4],
#     3: [1],
#     4: [1, 2]
# }
# print(dfs(graph, 0))

# Breadth-First Search (BFS)

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

# Example usage
# graph = {
#     0: [1, 2],
#     1: [0, 3, 4],
#     2: [0, 4],
#     3: [1],
#     4: [1, 2]
# }
# print(bfs(graph, 0))

# [BACKTRACKING]
# Permutations
def permute(nums):
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if i in used:
                continue
            used.add(i)
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used.remove(i)

    result = []
    used = set()
    backtrack([])
    return result

# Example usage
# nums = [1, 2, 3]
# print(permute(nums))
