'''
283. 移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        n = 0
        for i in range(len(nums)):
            if nums[i]:
                nums[i], nums[n] = nums[n], nums[i]
                n += 1


3502. 到达每个位置的最小费用
给你一个长度为 n 的整数数组 cost 。当前你位于位置 n（队伍的末尾），队伍中共有 n + 1 人，编号从 0 到 n 。
你希望在队伍中向前移动，但队伍中每个人都会收取一定的费用才能与你 交换位置。与编号 i 的人交换位置的费用为 cost[i] 。
你可以按照以下规则与他人交换位置：
如果对方在你前面，你 必须 支付 cost[i] 费用与他们交换位置。
如果对方在你后面，他们可以免费与你交换位置。
返回一个大小为 n 的数组 answer，其中 answer[i] 表示到达队伍中每个位置 i 所需的 最小 总费用。
class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        for i in range(1, len(cost)):
            cost[i] = min(cost[i - 1], cost[i])
        return cost


3541. 找到频率最高的元音和辅音
给你一个由小写英文字母（'a' 到 'z'）组成的字符串 s。你的任务是找出出现频率 最高 的元音（'a'、'e'、'i'、'o'、'u' 中的一个）和出现频率最高的辅音（除元音以外的所有字母），并返回这两个频数之和。
注意：如果有多个元音或辅音具有相同的最高频率，可以任选其中一个。如果字符串中没有元音或没有辅音，则其频率视为 0。
一个字母 x 的 频率 是它在字符串中出现的次数。
class Solution:
    def maxFreqSum(self, s: str) -> int:
        mp = Counter(s)
        vowel = max((mp[ch] for ch in mp if ch in "aeiou"), default=0)
        consonant = max((mp[ch] for ch in mp if ch not in "aeiou"), default=0)
        return vowel + consonant
'''
