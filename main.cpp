#include <iostream>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stack>
#include <queue>
#include <set>

using namespace std;

//t1 整数除法
int divideCore(int dividend, int divisor) {
    int result = 0;
    while (dividend >= divisor) {
        int value = divisor;
        int quotient = 1;
        while (value >= 0xc0000000 && dividend >= value * 2) {
            quotient *= 2;
            value *= 2;
        }
        result += quotient;
        dividend -= value;
    }

    return result;

}

int divide(int dividend, int divisor) {
    if (dividend == INT_MIN && divisor == -1) {
        return INT_MAX;
    }

    int negative = 2;
    if (dividend > 0) {
        negative--;
        dividend = -dividend;
    }

    if (divisor > 0) {
        negative--;
        dividend = -dividend;
    }
    int result = divideCore(dividend, divisor);

    return negative == 1 ? -result : result;
}

//t2 二进制加法
string addBinary(string a, string b) {
    string result = "";
    int i = a.length() - 1;
    int j = b.length() - 1;
    int carry = 0;
    while (i >= 0 || j >= 0) {
        int digitA = i >= 0 ? a.at(i--) - '0' : 0;
        int digitB = j >= 0 ? b.at(j--) - '0' : 0;
        int sum = digitA + digitB + carry;
        carry = sum >= 2 ? 1 : 0;
        sum = sum >= 2 ? sum - 2 : sum;
        result += to_string(sum);
    }
    if (carry) result += to_string(1);
    reverse(result.begin(), result.end());
    return result;
}

//t3 前n个数字二进制形式中1的个数
int* countBits(int num) {
    int* result = new int[num + 1];
    memset(result,  0, sizeof(int) * 5);
    for (int i = 0; i <= num; ++i) {
        int j = i;
        while (j != 0) {
            result[i]++;
            j = j & (j - 1);
        }
    }

    return result;
}

//优化版本
int* countBitsOptimize(int num) {
    int * result = new int[num + 1];
    memset(result, 0 ,sizeof(int) * 4);
    for (int i = 1; i <= num; ++i) {
        result[i] = result[i & (i - 1)] + 1;
    }

    return result;
}


//t4 只出现一次的数字
int singleNumber() {
    int bitNums[32] = {0};
    vector<int> nums{1, 0, 1, 0, 1, 0, 100, 100};
    for (auto num : nums) {
        for (int i = 0; i < 32; i++) {
            bitNums[i] += (num >> (31 - i)) & 1;
        }
    }

    int result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) + bitNums[i] % 3;
    }

    return result;
}

//t5 单词长度的最大乘积
// 输入一个字符串数组words，请计算不包含相同字符的两个字符串words[i]和words[j]长度乘积的最大值，如果没有则返回0
int maxProduct() {
    string words[5] = {"abcw", "foo", "bar", "fxyz", "abcdef"};
    unordered_map<string, unordered_map<char, bool>> hash;
    for (auto word : words) {
        for (auto ch : word) {
            hash[word][ch] = true;
        }
    }

    int result = 0;
    for (int i = 0; i < words->length(); i++) {
        for (int j = i + 1; j < words->length(); j++) {
            char k = 'a';
            for (; k <= 'z'; ++k) {
                if (hash[words[i]][k] == true && hash[words[j]][k] == true) {
                    break;
                }
            }
            if (k == 'z' + 1){
                int prod = words[i].length() * words[j].length();
                result = max(result, prod);
            }

        }
    }

    return result;
}

int maxProduct2() {
    string words[5] = {"abcw", "foo", "bar", "fxyz", "abcdef"};
    int flags[5];
    memset(flags, 0, sizeof(int) * 4 );
    for (int i = 0; i < words->length(); i++) {
        for (int j = 0; j < words[i].length(); j++) {
            flags[i] |= 1 << (words[i][j] - 'a');
        }
    }

    int result = 0;
    for (int i = 0; i < words->length(); i++) {
        for (int j = i + 1; j < words->length(); j++) {
            if ((flags[i] & flags[j]) == 0) {
                int prod = words[i].length() * words[j].length();
                result = max(prod, result);
            }
        }
    }

    return result;
}


/*
 * 第二章 数组
 */

pair<int, int> twoSum(vector<int> &numbers, int target) {
    int i = 0, j = numbers.size() - 1;
    while (i < j && numbers[i] + numbers[j] != target) {
        if (numbers[i] + numbers[j] < target) {
            i++;
        } else {
            j--;
        }
    }

    return make_pair(i, j);
}


// t7 数组中和为0的三个数字
// 输入一个数组，找出数组中所有和为0的三个数字的三元组

void twoSum(vector<int>& nums, int i, vector<vector<int>>& result) {
    int j = i + 1;
    int k = nums.size() - 1;
    while (j < k) {
        if (nums[i] + nums[k] + nums[j] == 0) {
            result.push_back({nums[i], nums[j], nums[k]});

            int temp = nums[j];
            while (nums[j] == temp && j < k) {
                ++j;
            }
        } else if (nums[i] + nums[k] + nums[j] < 0) {
            ++j;
        } else {
            --k;
        }
    }
}
vector<vector<int>> threeSum() {
    vector<int> nums{-1, 0, 1, 2, -1, -4,5};
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    int i = 0;
    while (i < nums.size() - 2) {
        twoSum(nums, i, result);
        int temp = nums[i];
        while (temp == nums[i] && i < nums.size() - 2) {
            ++i;
        }
    }

    return result;
}


//t8 和大于等于k的最短连续子数组

int minSubArrayLen(int k, vector<int>& nums) {
    int left = 0, right = 0;
    int sum = 0;
    int minLen = INT_MAX;
    for (; right < nums.size(); ++right) {
        sum += nums[right];
        while (left <= right && sum >= k) {
            minLen = min(minLen, right - left +1);
            sum -= nums[left++];
        }
    }

    return minLen == INT_MAX ? 0 : minLen;
}


//t9 乘积小于K的最短连续子数组
int numSubarrayProductLessThanK(vector<int> & nums, int k) {
    long product = 1;
    int left = 0;
    int count = 0;
    for (int right = 0; right < nums.size(); right++) {
        product *= nums[right];
        while (left <= right && product >= k) {
            product /= nums[left++];
        }
        count += right >= left ? right - left + 1 : 0;
    }

    return count;
}


//t10 输入一个整数数组和一个数字K，请问数组中有多少个数字之和等于k的连续子数组？
int subarraySum(vector<int> & nums, int k) {
    unordered_map<int, int> sumToCount;
    sumToCount.insert(make_pair(0,1)); //这是给从下标从0开始的那段用
    int sum = 0;
    int count = 0;
    for (int num : nums) {
        sum += num;
        if (sumToCount.count(sum-k)){
            count += sumToCount.at(sum - k);
        }
        sumToCount[sum]++;
    }

    return count;
}


//t11 0和1个数相等最长连续的子数组
int findMaxLength(vector<int> & nums) {
    int maxLength = INT_MIN;
    int sum = 0;
    unordered_map<int, int> hash;
    hash.insert(make_pair(0, -1)); //
    for (int i = 0;i < nums.size(); i++) {
        sum += nums[i] == 0 ? -1 : 1;
        if (hash.count(sum)) {
            maxLength = max(maxLength, i - hash[sum]);
        } else {
            hash.insert(make_pair(sum, i));
        }
    }

    return maxLength;
}

//t12 左右两边子数组的和相等
int pivotIndex(vector<int> & nums) {
    int total = 0;
    for (auto it : nums ) {
        total += it;
    }

    int sum = 0;
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
        if (sum - nums[i] == total - sum) {
            return i;
        }
    }

    return -1;
}


//t13 二维子数组之和
//输入一个二维矩阵，如何计算给定左上角坐标和右下角坐标的子矩阵数字之和？
class NumMatrix {
private:
    int sums[100][100] = {{}};

public:
    NumMatrix(vector<vector<int>> matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return ;
        }

        for (int i = 0;i < matrix.size(); i++) {
            int rowSum = 0;
            for (int j = 0; j < matrix[0].size(); j++) {
                rowSum += matrix[i][j];
                sums[i+1][j+1] = sums[i][j+1] + rowSum;
            }
        }
     }

    int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1 ] - sums[row2 + 1][col1]
               -sums[row1][col2 + 1] + sums[row1][col1];
    }
};


/*
 * 第三章字符串
 */

//3.2 双指针
//t14 字符串中的变位词
bool areAllZero(vector<int> & counts) {
    for (int count : counts) {
        if (count != 0) {
            return false;
        }
    }

    return true;
}

bool checkinclusion(string s1, string s2) {
    if (s1.size() > s2.size()) {
        return false;
    }

    vector<int> counts(26, 0);
    for (int i = 0; i < s1.size(); i++) {
        counts[s1[i] - 'a']++;
        counts[s2[i] - 'a']--;
    }

    if (areAllZero(counts)) {
        return true;
    }

    for (int i = s1.size(); i < s2.size(); i++) {
        counts[s2[i] - 'a']--;
        counts[s2[i - s1.size()] - 'a']++;
        if (areAllZero(counts)) {
            return true;
        }
    }

    return false;
}

//t15 字符串中的所有变位词
//时间复杂度O(n),空间复杂度O(1)
//调用示例
//for(auto it: findAnagrams("abc", "cbadabacg")) {
//    cout << it << " ";
//}
vector<int> findAnagrams(string s1, string s2) {
    if (s1.size() > s2.size()) {
        return {};
    }

    vector<int> result;
    vector<int> counts(26, 0);
    for (int i = 0; i < s1.size(); i++) {
        counts[s1[i] - 'a']++;
        counts[s2[i] - 'a']--;
    }

    if (areAllZero(counts)) {
        result.push_back(0);
    }

    for (int i = s1.size(); i < s2.size(); i++) {
        counts[s2[i] - 'a']--;
        counts[s2[i - s1.size()] - 'a']++;
        if (areAllZero(counts)) {
            result.push_back(i - s1.size() + 1);
        }
    }

    return result;
}

//t16 不含重复字符的最长子字符串
//调用案例

bool hasMoreThan1(vector<int> & counts) {
    for (auto count : counts) {
        if (count > 1) {
            return true;
        }
    }

    return false;
}

int lengthOfLongestSubstring(string s) {
    if (s.length() == 0) {
        return 0;
    }
    int maxLen = INT_MIN;
    int j = 0;
    vector<int> alphabet(26, 0);
    for (int i = 0; i < s.size(); i++) {
        alphabet[s.at(i) - 'a']++;
        while (hasMoreThan1(alphabet)) {
            alphabet[s.at(j++) - 'a']--;
        }
        maxLen = max(maxLen, i - j + 1);
    }

    return maxLen;
}



int lengthOfLongestSubstringOptimism(string s) {
    if (s.length() == 0) {
        return 0;
    }
    int maxLen = INT_MIN;
    int j = 0;
    vector<int> alphabet(26, 0);
    int countDup = 0;
    for (int i = 0; i < s.size(); i++) {
        alphabet[s.at(i) - 'a']++;
        if (alphabet[s[i] - 'a'] == 2) {
            countDup++;
        }
        while (countDup > 0) {
            alphabet[s[j++] - 'a']--;
            if (alphabet[s[i] - 'a'] == 1) { //此处有勘误， j 应该替换为 i
                countDup--;
            }
        }
        maxLen = max(maxLen, i - j + 1);
    }

    return maxLen;
}

//t17 包含所有字符的最短字符串
//cout << minWindow("addbancad", "abc"); 测试样例
string minWindow(string s, string t) {
    unordered_map<char, int> hash;
    for (auto it : t) {
        hash[it]++;
    }

    int count = hash.size();
    cout << "hash_size : " << count << endl;
    int start = 0;
    int end = 0;
    int minStart = 0, minEnd = 0;
    int minLen = INT_MAX;
    while (end < s.size() || (count == 0 && end == s.size())) { //改成end <= s.size()也可
        if (count > 0) {
            if (hash.count(s[end])) {
                hash[s[end]]--;
                if (hash[s[end]] == 0) {
                    count--;
                }
            }
            end++;
        } else {
            if (minLen > end - start) {
                minLen = end - start;
                minStart = start;
                minEnd = end;
            }

            if (hash.count(s[start])) {
                hash[s[start]]++;
                if (hash[s[start]] == 1 ) {
                    count++;
                }
            }

            start++;
        }
    }

    return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
}


//3.2 回文字符串
//t18 有效的回文（忽略大小写或者标点符号）
//时间复杂度O（n)
//cout << isPalindrome("qwer,r,e,w,q");
bool isPalindrome(string s) {
    int i = 0;
    int j = s.size() - 1;
    while (i < j) {
        char ch1 = s[i];
        char ch2 = s[j];
        if (!isalnum(ch1)) {
            i++;
        } else if(!isalnum(ch2)) {
            j--;
        } else {
            if (tolower(ch1) != tolower(ch2)) {
                return false;
            }

            i++;
            j--;
        }
    }
    return true;
}

//t19 最多删除一个字符得到回文
//cout << validPalindrome("abcdecba");
bool isPalindrome(string s, int start, int end) {
    while (start < end) {
        if (s[start] != s[end]) {
            break;
        }
        start++;
        end--;
    }


    return start >= end;
}
bool validPalindrome(string s) {
    int start = 0;
    int end = s.size() - 1;
    for (; start < s.size() / 2; ++start, --end) {
        if (s[start] != s[end]) {
            break;
        }
    }

    return start == s.size() / 2 ||
           isPalindrome(s, start + 1, end) ||
           isPalindrome(s, start, end - 1);
}

//t20 回文子字符串的个数
//时间复杂度O（n2), 空间O(1)
//cout << countSubStrings("aaaa");
int countPalindrome(string s, int start, int end) {
    int count = 0;
    while (start >= 0 && end < s.length() && s[start] == s[end]) {
        count++;
        start--;
        end++;
    }

    return count;
}
int countSubStrings(string s) {
    if (s.size() == 0) {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < s.length(); ++i) {
        count += countPalindrome(s, i, i);
        count += countPalindrome(s, i, i + 1);
    }

    return count;
}

/*
 * 第五章 哈希表
 */



//t30 插入、删除和随机访问都是O(1)的容器
/*
 * RandomizedSet randomizedSet;
    cout << randomizedSet.insert(5);
    cout << randomizedSet.insert(6);
    cout << randomizedSet.insert(7);
    cout << randomizedSet.insert(7);
    cout << randomizedSet.erase(7);
 */
class RandomizedSet {
private:
    unordered_map<int, int> numToLocation;
    vector<int> nums;

public:
    bool insert(int val) {
        if (numToLocation.count(val)) {
            return false;
        }

        numToLocation.insert(make_pair(val, nums.size()));
        nums.push_back(val);

        return true;
    }

    bool erase(int val) {
        if (!numToLocation.count(val)) {
            return false;
        }

        int location = numToLocation.at(val);
        numToLocation.insert(make_pair(nums.back(), location));
        numToLocation.erase(val);
        nums[location] = nums.back();
        nums.pop_back();
        return true;
    }

    int getRandom() {
        return nums[rand() % nums.size()];
    }
};

//t31 LRU
class LRUCache {
public:
    LRUCache(int capacity) {
        _capacity = capacity;
    }

    int get(int key) {
        auto it = hash.find(key);
        if (it == hash.end()) {
            return -1;
        }
        cache.splice(cache.begin(), cache, it->second);
        return it->second->second;

    }

    void put(int key, int value) {
        auto it = hash.find(key);
        if (it != hash.end()) {
            it->second->second = value;
            cache.splice(cache.begin(), cache, it->second);
            //hash[key] = cache.begin();
            return;
        }

        if (_capacity == cache.size()) {
            auto node = cache.back();
            hash.erase(node.first);
            cache.pop_back();
        }

        cache.emplace_front(make_pair(key, value));
        hash[key] = cache.begin();

    }
private:
    int _capacity;
    list<pair<int, int>> cache;
    unordered_map<int, list<pair<int, int>>::iterator> hash;
};

//T32 变位词
//cout << isAnagram("abcde", "edcba");
//时间O(n), 空间O(n)
bool isAnagram(string s1, string s2) {
    if (s1.size() != s2.size()) {
        return false;
    }

    unordered_map<int, int> counts;
    for (char ch : s1) {
        counts[ch]++;
    }

    for (char ch : s2) {
        if (!counts.count(ch) || counts[ch] == 0) {
            return false;
        }
        counts[ch]--;
    }

    return true;
}

//t33 变位词组
/*
 * vector<string> strs = {"eat", "ate", "eta", "sl", "aaa", "ls"};
    vector<vector<string>> result = groupAnagrams(strs);
    for (auto vec : result) {
        for (auto str : vec) {
            cout << str << " ";
        }
    }
 */
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    for (string str : strs) {
        string temp = str;
        sort(temp.begin(), temp.end());
        if (!groups.count(temp)) {
            vector<string> vec;
            groups.insert(make_pair(temp, vec));
            groups[temp].push_back(str);
        } else {
            groups[temp].push_back(str);
        }

    }

    vector<vector<string>> result;
    for (auto [k1, v1] : groups) {
        result.push_back(v1);
    }

    return result;
}


//t34 外形语言是否排序
//输入一组单词["offer", "is", "coming] 以及字母表顺序"zyxw...a"
//
bool isSorted(string word1, string word2, int orderArray[]) {
    int i = 0;
    for (; i < word1.size() && i < word2.size(); i++) {
        char ch1 = word1[i];
        char ch2 = word2[i];
        if (orderArray[ch1 - 'a'] < orderArray[ch2 - 'a']) {
            return true;
        }

        if (orderArray[ch1 - 'a'] > orderArray[ch2 - 'a']) {
            return false;
        }
    }

    return i == word1.size();
}

bool isAlienSorted(vector<string>& words, string order) {
    int orderArray[26] = {};
    for (int i = 0; i < words.size() ; i++) {
        orderArray[order[i] - 'a'] = i;
    }

    for (int i = 0; i < words.size() - 1; i++) {
        if (!isSorted(words[i], words[i+1], orderArray)) {
            return false;
        }
    }

    return true;
}

//t35 最小时间差
//vector<string> timePoints = {"23:50", "23:59", "00:00", "00:01"};
//    cout << findMinDifference(timePoints);
int helper(bool minuteFlags[]) {
    int minDiff = INT_MAX;
    int prev = -1;
    int first = INT_MAX;
    int last = -1;
    for (int i = 0; i < 1440; i++) {
        if (minuteFlags[i]) {
            if (prev >= 0) {
                minDiff = min(i - prev, minDiff);
            }

            prev = i;
            first = min(i, first);
            last = max(i, last);
        }
    }

    minDiff = min(first + 1440 - last, minDiff);
    return minDiff;
}
int findMinDifference(vector<string>& timePoints) {
    if (timePoints.size() > 1440) {
        return 0;
    }

    bool minuteFlags[1440] = {};
    for (string time : timePoints) {
        int pos = time.find(":");
        int minute = stoi(time.substr(0, pos)) * 60 + stoi(time.substr(pos + 1, 2));
        if (minuteFlags[minute]) {
            return 0;
        }
        minuteFlags[minute] = true;
    }

    return helper(minuteFlags);
}

//t36 后缀表达式
//时间O(n), 空间O(n)
// vector<string> v = {"2", "1", "3", "*", "+"};
//    cout << evalRPN(v);
int calculate(int num1, int num2, string& opera) {
    if (opera == "+") {
        return num1 + num2;
    }

    if (opera == "-") {
        return num1 - num2;
    }

    if (opera == "*") {
        return num1 * num2;
    }

    if (opera == "/") {
        return num1 / num2;
    }

    return 0;
}

int evalRPN(vector<string> & tokens) {
    stack<int> stk;
    for (string token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/"){
            int num1 = stk.top(); stk.pop();
            int num2 = stk.top(); stk.pop();
            stk.push(calculate(num2, num1, token));
        } else {
            stk.push(stoi(token));
        }
    }

    return stk.top();

}

//t37 小行星碰撞
//时间O(n), 空间O(N)
//vector<int> v = {4, 5, -6, 4, 8, -5};
//    for (auto it : asteroidCollision(v)) {
//        cout << it << " ";
//    }
vector<int> asteroidCollision(vector<int> & asteroids) {
    stack<int> stack;
    for (auto as : asteroids) {
        while (!stack.empty() && stack.top() > 0 && stack.top() < -as) {
            stack.pop();
        }

        if (!stack.empty() && stack.top() == -as && as < 0) {
            stack.pop();
        } else if (as > 0 || stack.empty() || stack.top() < 0) {
            stack.push(as);
        }
    }

    vector<int> v;
    while (!stack.empty()) {
        v.push_back(stack.top());
        stack.pop();
    }
    reverse(v.begin(), v.end());
    return v;
}



//t38 每日温度
//时间复杂度O(n), 空间复杂度O(n)
//vector<int> v = {35, 31, 33, 36, 34};
//print(dailyTemperatures(v));
vector<int> dailyTemperatures(vector<int> temperatures) {
    stack<int> stack;
    vector<int> result(temperatures.size(), 0);
    for (int i = 0; i < temperatures.size(); i++) {
        while (!stack.empty() && temperatures[i] > temperatures[stack.top()]) {
            int prev = stack.top();
            stack.pop();
            result[prev] = i - prev;
        }
        stack.push(i);
    }


    return result;
}

void print(vector<int> v) {
    for (auto it : v) {
        cout << it << " ";
    }
}


//t39 直方图最大矩形面积

//1暴力法， O（n2)
int largestRectangleArea(vector<int>& heights) {
    int maxArea = 0;
    for (int i = 0; i < heights.size(); i++) {
        int minHeight = heights[i];
        for (int j = i; j < heights.size(); j++) {
            minHeight = min(minHeight, heights[j]);
            maxArea = max(minHeight * (j - i + 1), maxArea);
        }
    }

    return maxArea;
}

//2 单调栈
//时间O(N), 空间O(N)
//    vector<int> v = {3, 2, 5, 4, 6, 1, 4, 2};
//    cout << largestRectangleArea2(v);
int largestRectangleArea2(vector<int> & heights) {
    int maxArea = 0;
    stack<int> stack;
    for (int i = 0; i < heights.size(); i++) {
        while (!stack.empty() && heights[stack.top()] >= heights[i]) {
            int curHeight = heights[stack.top()];
            stack.pop();
            if (!stack.empty()) {
                maxArea = max(curHeight * (i - stack.top() - 1), maxArea);
            } else {
                maxArea = max(curHeight * (i - (-1) - 1 ), maxArea);
            }

        }

        stack.push(i);
    }

    while (!stack.empty()) {
        int curHeight = heights[stack.top()];
        stack.pop();
        if (!stack.empty()) {
            maxArea = max(curHeight * (int(heights.size()) - stack.top() - 1), maxArea);
        } else {
            maxArea = max(curHeight * (int(heights.size()) - (-1) - 1 ), maxArea);
        }
    }

    return maxArea;
}

//40 矩阵中的最大矩形
//vector<vector<char>> v = {
//        {'1', '0', '1', '0', '0'},
//        {'0', '0', '1', '1', '1'},
//        {'1', '1', '1', '1', '1'},
//        {'1', '0', '0', '1', '0'}
//
//};
//cout << maximalRectangle(v);
int maximalRectangle(vector<vector<char>> matrix) {
    if (matrix.size() == 0 || matrix[0].size() == 0) {
        return 0;
    }

    vector<int> heights(matrix[0].size(), 0);
    int maxArea = 0;
    for (auto row : matrix) {
        for (int i = 0; i < row.size(); i++) {
            if (row[i] == '0') {
                heights[i] = 0;
            } else {
                heights[i]++;
            }
        }

        maxArea = max(maxArea, largestRectangleArea2(heights));
    }

    return maxArea;
}


//第九章 堆
//59 数据流的第K大数字
class KthLargest {
private:
    priority_queue<int, vector<int>, greater<int>> minHeap; //小顶部堆
    int size;
public:
    KthLargest(int k, vector<int> &v) {
        size = k;
        for (int num : v) {
            add(num);
        }
    }

    int add(int val) {
        if (minHeap.size() < size) {
            minHeap.push(val);
        } else if (minHeap.top() < val) {
            minHeap.pop();
            minHeap.push(val);
        }

        return minHeap.top();
    }
};


//t60 出现频率最高的k个数字
class cmp{
public:
    bool operator() (pair<int, int> m1, pair<int, int> m2) {
        return m1.second < m2.second;
    }
};
vector<int> topKFrequent(vector<int>& nums, int k){
    unordered_map<int, int> hash;
    for (int num : nums) {
        hash[num]++;
    }

    priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> minHeap;

    for (pair<int, int> node : hash) {
        if (minHeap.size() < k) {
            minHeap.push(node);
        } else if (node.second > minHeap.top().second) {
            minHeap.pop();
            minHeap.push(node);
        }
    }

    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().first);
        minHeap.pop();
    }
    return result;
}

//和最小的k个数对
//vector<int> nums1 = {1, 5, 13, 21};
//    vector<int> nums2 = {2, 4, 9, 15};
//    vector<vector<int>> res =  kSmallPairs(nums1, nums2, 3);
//    for (auto num : res) {
//        cout << "(" << num[0] << " " << num[1] << ")" << endl;
//    }
class cmp2 {
public:
    bool operator()(pair<int, int> p1, pair<int, int> p2) {
        return p1.second + p1.first > p2.second + p2.first;
    }
};
vector<vector<int>> kSmallPairs(vector<int> nums1, vector<int> nums2, int k) {
    priority_queue<pair<int, int>, vector<pair<int, int>>, cmp2> maxHeap;
    for (int i = 0; i < min(int(nums1.size()), k); i++) {
        for (int j = 0; j < min(int(nums2.size()), k); j++) {
            if (maxHeap.size() >= k) {
                int curNum = maxHeap.top().first + maxHeap.top().second;
                if (curNum > nums1[i] + nums2[j]) {
                    maxHeap.pop();
                    maxHeap.push(make_pair(nums1[i], nums2[j]));
                }
            } else {
                maxHeap.push(make_pair(nums1[i], nums2[j]));
            }
        }
    }
    vector<vector<int>> result;
    while (!maxHeap.empty()) {
        result.push_back({maxHeap.top().first, maxHeap.top().second});
        maxHeap.pop();
    }

    return result;
}

//t68 查找插入位置
//vector<int> nums = {1, 3, 6, 8};
//cout << searchInsert(nums, 5);
int searchInsert(vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (nums[mid] >= target) {
            if (mid == 0 || nums[mid - 1] < target) {
                return mid;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return nums.size();
}

//t69 山峰数组的顶部
//vector<int> v = {1, 3, 5, 4, 2};
//    cout << peakIndexInMountainArray(v);
int peakIndexInMountainArray(vector<int> & nums) {
    int left = 1;
    int right = nums.size() - 2;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
            return mid;
        }

        if (nums[mid] > nums[mid - 1]) {
            left = mid + 1;
        } else {
            right = mid -1;
        }
    }

    return -1;

}

//t70 排序数组中只出现一次的数字
//vector<int> v = {1, 1, 2, 2, 3, 4, 4};
//    cout << singleNonDuplicate(v);
int singleNonDuplicate(vector<int> & nums) {
    int left = 0;
    int right = nums.size() / 2;
    while (left <= right) {
        int mid = (left + right) / 2;
        int i = mid * 2;
        if (i < nums.size() - 1 && nums[i] != nums[i + 1]){
            if (mid == 0 || nums[i - 2] == nums[i - 1]) {
                return nums[i];
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }

    }

    return nums[nums.size() - 1];
}

//t71 按权重生成随机数
//vector<int> sums = {1, 2, 3, 4};
//    Solution s(sums);
//    for (int i = 0; i < 10; i++) {
//        cout << s.pickIndex() << " ";
//    }
class Solution {
private:
    vector<int> sums;
    int total = 0;
public:
    Solution(vector<int>& w) {
        for(int i = 0; i < w.size(); i++) {
            total += w[i];
            sums.push_back(total);
        }
    }

    int pickIndex() {
        int p = rand() % total;
        int left = 0;
        int right = sums.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (sums[mid] > p) {
                if (mid == 0 || p >= sums[mid - 1]) {
                    return p;
                }

                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return -1;
    }
};


//t72求平方根
int mySort(int n) {
    int left = 1;
    int right = n;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (mid <= n / mid) {
            if (mid + 1 >= n / (mid + 1)) {
                return mid;
            }
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

//t73 狒狒吃香蕉
//vector<int> piles = {3, 6, 7, 11};
//    cout << minEatingSpeed(piles, 8);
//如果总共有m堆香蕉， 最大一堆的香蕉数目为n， 则效率为O(mlogn)
int getHours(vector<int>& values, int speed) {
    int hour = 0;
    for (auto it : values) {
        hour += (it + speed - 1) / speed;
    }

    return hour;
}

int minEatingSpeed(vector<int>& values, int hour) {
    int maxSpeed = INT_MIN;
    for (auto it : values) {
        maxSpeed = max(maxSpeed, it);
    }
    int left = 1;
    int right = maxSpeed;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (getHours(values, mid) <= hour) {
            if (mid == 1 || getHours(values, mid - 1) > hour) {
                return mid;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return -1;
}


//第十二章 排序
//t74 合并区间
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(), [=](vector<int> a, vector<int> b)      {
        return a[0] < b[0];
    });

    vector<vector<int>> result;
    int i = 0;
    while (i < intervals.size()) {
        vector<int> temp = {intervals[i][0], intervals[i][1]};
        int j = i + 1;
        while (j < intervals.size() && temp[1] >= intervals[j][0]) {
            temp[1] = max(temp[1], intervals[j][1]);
            j++;
        }

        result.push_back(temp);
        i = j;
    }

    return result;

}

//t75 数组相对排序
vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
    unordered_map<int, int> hash;
    for (auto it : arr1) {
        hash[it]++;
    }

    int index = 0;
    for (auto it : arr2) {
        while (hash.count(it) && hash[it] > 0) {
            arr1[index++] = it;
            hash[it]--;
        }
    }

    for (int i = 0; i <= 1000; i++) {
        while (hash[i] > 0) {
            arr1[index++] =  i;
            hash[i]--;
        }
    }

    return arr1;
}

//第十三章回溯法
//t79所有子集
//O(2的n次方）
void helper(vector<int>& nums, int index, vector<int>& subset, vector<vector<int>>& result) {
    if (index == nums.size()) {
        result.push_back(subset);
    } else if (index < nums.size()) {
        helper(nums, index + 1, subset, result);

        subset.push_back(nums[index]);
        helper(nums, index + 1, subset, result);
        subset.pop_back();
    }
}

vector<vector<int>> subsets(vector<int> nums) {
    vector<vector<int>> result;
    if (nums.size() == 0) {
        return result;
    }

    vector<int> subset;
    helper(nums, 0, subset, result);
    return result;
}

ostream& operator<<(ostream& cout, vector<vector<int>> v) {
    for (auto vec : (v)) {
        if (vec.empty()) {
            cout << "null" << endl;
            continue;
        }
        for (auto it : vec) {
            cout << it << " ";
        }
        cout << endl;
    }

    return cout;
}

//t80 包含K个元素的集合
void helper(int n, int i, int k, vector<int>& combination, vector<vector<int>>& result) {
    if (combination.size() == k) {
        result.push_back(combination);
    } else if(i <= n) {
        helper(n, i + 1, k, combination, result);
        combination.push_back(i);
        helper(n, i + 1, k, combination, result);
        combination.pop_back();
    }
}
vector<vector<int>> combine(int n, int k) {
    if (n == 0 || k == 0) {
        return {{}};
    }

    vector<vector<int>> result;
    vector<int> combination;
    helper(n, 1, k, combination, result);

    return result;
}


//t81 允许重复选择元素的组合
void helper(vector<int>& nums, int target, int i, vector<int>& combination, vector<vector<int>>& result) {
    if (target == 0) {
        result.push_back(combination);
    } else if (target > 0 && i < nums.size()) {
        helper(nums, target, i + 1, combination, result);

        combination.push_back(nums[i]);
        helper(nums, target - nums[i], i, combination, result);
        combination.pop_back();
    }
}

vector<vector<int>> combinationSUm(vector<int> nums,int target) {
    vector<vector<int>> result;
    if (nums.size() == 0) {
        return result;
    }

    vector<int> combination;
    helper(nums, target, 0, combination, result);
    return result;
}


//t83 没有重复元素集合的全排列
void helper(vector<int>& nums, int i, vector<vector<int>>& result) {
    if (i == nums.size()) {
        vector<int> permutation(nums.begin(), nums.end());
        result.push_back(permutation);
    } else {
        for (int j = i; j < nums.size(); j++) {
            swap(nums[i], nums[j]);
            helper(nums, i + 1, result);
            swap(nums[i], nums[j]);
        }
    }
}
 vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    helper(nums, 0, result);
    return result;
}

//t84 包含重复元素集合的全排列
void helper2(vector<int>& nums, int i, vector<vector<int>>& result) {
    if (i == nums.size()) {
        vector<int> permutation(nums.begin(), nums.end());
        result.push_back(permutation);
    } else {
        set<int> set;
        for (int j = i; j < nums.size(); j++) {
            if (!set.count(nums[j])) {
                set.insert(nums[j]);

                swap(nums[i], nums[j]);
                helper2(nums, i + 1, result);
                swap(nums[i], nums[j]);
            }
        }
    }
}
vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> result;
    helper2(nums, 0, result);
    return result;

}

//t86 分割回文子字符串
void helper(string str, int start, vector<string> substrings, vector<vector<string>>& result) {
    if (start == str.size()) {
        result.push_back(substrings);
        return;
    }

    for (int i = start; i < str.size(); i++) {
        if (isPalindrome(str, start, i)) {
            substrings.push_back(str.substr(start, i - start + 1));
            helper(str, i + 1, substrings, result);
            substrings.pop_back();
        }
    }
}

vector<vector<string>> partition(string s) {
    vector<vector<string>> result;
    vector<string> subStrings;
    helper(s, 0, subStrings, result);

    return result;
}


//t87 恢复IP地址
bool isValidSeg(string seg) {
    return stoi(seg) <= 255 && (seg == "0" || seg[0] != '0');
}

void helper(string s, int i, int segI, string seg, string ip, vector<string>& result) {
    if (i == s.size() && segI == 3 && isValidSeg(seg)) {
        result.push_back(ip + seg);
    } else if (i < s.size() && segI <= 3) {
        char ch = s[i];
        if (isValidSeg(seg + ch)) {
            helper(s, i + 1, segI, seg + ch, ip, result);
        }

        if (seg.size() > 0 && segI < 3) {
            helper(s, i + 1, segI + 1,  string("") + ch, ip + seg + ".", result);
        }
    }
}

vector<string> restoreIpAddress(string s) {
    vector<string> result;
    helper(s, 0, 0, "", "", result);

    return result;
}

//第十四章动态规划

//t88爬楼梯的最少成本
//大量的重复计算，并不好
int helper(vector<int>& cost, int i) {
    if (i < 2) {
        return cost[i];
    }

    return min(helper(cost, i - 2), helper(cost, i - 1)) + cost[i];
}

int minCostClimbStairs(vector<int>& cost) {
    int len = cost.size();
    return min(helper(cost, len - 2), helper(cost, len - 1));
}

//使用缓存的递归代码

int helper(vector<int>& cost, int i, unordered_map<int, int>& hash) {
    if (i < 2) {
        return cost[i];
    } else if(hash.count(i)) {
        return hash[i];
    } else {
        int temp = min(helper(cost, i - 2), helper(cost, i - 1)) + cost[i];
        hash[i] = temp;
        return temp;
    }

}

int minCostClimbStairsBuffer(vector<int>& cost) {
    unordered_map<int, int> hash;
    int len = cost.size();
    return min(helper(cost, len - 2, hash), helper(cost, len - 1, hash));
}


//dp写法
int minCostClimbStairsDp(vector<int>& cost) {
    int len = cost.size();
    vector<int> dp(len);
    dp[0] = cost[0];
    dp[1] = cost[1];

    for (int i = 2; i < len; i++) {
        dp[i] = min(dp[i - 2], dp[i - 1]) + cost[i];
    }

    return min(dp[len - 1], dp[len - 2]);
}

//O(1) DP
int minCostClimbStairsDpZoneOptimism(vector<int>& cost) {
    int len = cost.size();
    int dp[2];
    dp[0] = cost[0];
    dp[1] = cost[1];

    for (int i = 2; i < len; i++) {
        dp[i % 2] = min(dp[0], dp[1]) + cost[i];
    }

    return min(dp[0], dp[1]);
}

// t89 打家劫舍
int rob(vector<int>& nums) {
    if (nums.size() == 0) {
        return 0;
    }

    int dp[2];
    dp[0] = nums[0];
    dp[1] = nums[1];

    for (int i = 2; i < nums.size(); i++) {
        dp[i % 2] = max(dp[(i - 1) % 2], dp[(i - 2) % 2]  + nums[i]);
    }

    return dp[(nums.size() - 1) % 2];
}

class MovingAverage {
public:
    /** Initialize your data structure here. */
    queue<int> q;
    int sum = 0;
    int size;
    MovingAverage(int size) {
        size = size;
    }

    double next(int val) {
        if (q.size() == size) {
            sum -= q.front();
            q.pop();
        }

        q.push(val);
        sum += val;
        return sum * 1.0 / q.size();
    }
};


int main() {
    cout << maxProduct();



}