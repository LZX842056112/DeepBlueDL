'''
3190. 使所有元素都可以被 3 整除的最少操作数
给你一个整数数组 nums 。一次操作中，你可以将 nums 中的 任意 一个元素增加或者减少 1 。
请你返回将 nums 中所有元素都可以被 3 整除的 最少 操作次数。
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return sum(x % 3 != 0 for x in nums)

535. TinyURL 的加密与解密
TinyURL 是一种 URL 简化服务， 比如：当你输入一个 URL https://leetcode.com/problems/design-tinyurl 时，它将返回一个简化的URL http://tinyurl.com/4e9iAk 。请你设计一个类来加密与解密 TinyURL 。
加密和解密算法如何设计和运作是没有限制的，你只需要保证一个 URL 可以被加密成一个 TinyURL ，并且这个 TinyURL 可以用解密方法恢复成原本的 URL 。
实现 Solution 类：
Solution() 初始化 TinyURL 系统对象。
String encode(String longUrl) 返回 longUrl 对应的 TinyURL 。
String decode(String shortUrl) 返回 shortUrl 原本的 URL 。题目数据保证给定的 shortUrl 是由同一个系统对象加密的。
class Codec:
    def __init__(self):
        self.map = {}
        self.id = 0

    def encode(self, longUrl: str) -> str:
        self.id += +1
        self.map[self.id] = longUrl
        return f"http://tinyurl.com/{str(self.id)}"

    def decode(self, shortUrl: str) -> str:
        i = shortUrl.rfind("/")
        id = int(shortUrl[i + 1 :])
        return self.map[id]

3668. 重排完成顺序
给你一个长度为 n 的整数数组 order 和一个整数数组 friends。
order 包含从 1 到 n 的每个整数，且 恰好出现一次 ，表示比赛中参赛者按照 完成顺序 的 ID。
friends 包含你朋友们的 ID，按照 严格递增 的顺序排列。friends 中的每个 ID 都保证出现在 order 数组中。
请返回一个数组，包含你朋友们的 ID，按照他们的 完成顺序 排列。
class Solution:
    def recoverOrder(self, order: List[int], friends: List[int]) -> List[int]:
        return[i for i in order if i in friends]
'''