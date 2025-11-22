'''
1828. 统计一个圆中点的数目
给你一个数组 points ，其中 points[i] = [xi, yi] ，表示第 i 个点在二维平面上的坐标。多个点可能会有 相同 的坐标。
同时给你一个数组 queries ，其中 queries[j] = [xj, yj, rj] ，表示一个圆心在 (xj, yj) 且半径为 rj 的圆。
对于每一个查询 queries[j] ，计算在第 j 个圆 内 点的数目。如果一个点在圆的 边界上 ，我们同样认为它在圆 内 。
请你返回一个数组 answer ，其中 answer[j]是第 j 个查询的答案。
class Solution:
    def countPoints(self, points: List[List[int]], queries: List[List[int]]) -> List[int]:
        res = []
        for q in queries:
            xj, yj, rj = q
            r_squared = rj * rj
            count = 0
            for p in points:
                xi, yi = p
                dx = xi - xj
                dy = yi - yj
                dist_squared = dx * dx + dy * dy
                if dist_squared <= r_squared:
                    count += 1
            res.append(count)
        return res

3168. 候诊室中的最少椅子数
给你一个字符串 s，模拟每秒钟的事件 i：
如果 s[i] == 'E'，表示有一位顾客进入候诊室并占用一把椅子。
如果 s[i] == 'L'，表示有一位顾客离开候诊室，从而释放一把椅子。
返回保证每位进入候诊室的顾客都能有椅子坐的 最少 椅子数，假设候诊室最初是 空的 。
class Solution:
    def minimumChairs(self, s: str) -> int:
        ans = cnt = 0
        for c in s:
            if c == 'E':
                cnt += 1
                ans = max(ans, cnt)
            else:
                cnt -= 1
        return ans

1769. 移动所有球到每个盒子所需的最小操作数
有 n 个盒子。给你一个长度为 n 的二进制字符串 boxes ，其中 boxes[i] 的值为 '0' 表示第 i 个盒子是 空 的，而 boxes[i] 的值为 '1' 表示盒子里有 一个 小球。
在一步操作中，你可以将 一个 小球从某个盒子移动到一个与之相邻的盒子中。第 i 个盒子和第 j 个盒子相邻需满足 abs(i - j) == 1 。注意，操作执行后，某些盒子中可能会存在不止一个小球。
返回一个长度为 n 的数组 answer ，其中 answer[i] 是将所有小球移动到第 i 个盒子所需的 最小 操作数。
每个 answer[i] 都需要根据盒子的 初始状态 进行计算。

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        count, total = 0, 0
        for j in range(len(boxes)):
            if boxes[j] == '1':
                count += 1
                total += j
        res = [0] * len(boxes)
        res[0] = total
        prev = total
        left = 0
        for i in range(1, len(boxes)):
            if boxes[i-1] == '1':
                left += 1
            right = count - left
            current = prev - right + left
            res[i] = current
            prev = current
        return res
'''
