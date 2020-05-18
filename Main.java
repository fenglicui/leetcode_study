import java.util.ArrayList;
import java.util.*;
import java.util.stream.Collectors;

import static java.util.Arrays.*;


public class Main {

    /**
     * LeetCode 21 从排序中删除重复项
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        int n = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[n++] = nums[i];
            }
        }
        return n;
    }

    /**
     * LeetCode 23 旋转数组
     * 循环移动
     *
     * @param nums
     * @param k
     */
    public static void rotate(int[] nums, int k) {
        if (k == 0) return;
        int count = 0, len = nums.length;
        for (int i = 0; i < len; i++) {
            if (count == len)
                break;
            int cur = i;
            int tmp = nums[cur];
            while (true) {
                if ((cur - k + len) % len == i)
                    break;
                nums[cur] = nums[(cur - k + len) % len];
                cur = (cur - k + len) % len;
                count++;
                if (count == len)
                    break;
            }
            nums[cur] = tmp;
            count++;
        }
    }

    //LeetCode 反转数组
    public static void reverse(int[] nums, int start, int end) {
        for (int i = start; i <= (end + start) / 2; i++) {
            int tmp = nums[i];
            nums[i] = nums[start + end - i];
            nums[start + end - i] = tmp;
        }
    }

    /**
     * LeetCode 旋转数组
     * 三次反转数组
     *
     * @param nums
     * @param k
     */
    public static void rotate1(int[] nums, int k) {
        int len = nums.length;
        k %= len;
        if (k == 0)
            return;
        reverse(nums, 0, len - k - 1);
        reverse(nums, len - k, len - 1);
        reverse(nums, 0, len - 1);
    }


    /**
     * LeetCode 24 存在重复
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (set.contains(i))
                return true;
            else set.add(1);
        }
        return false;
    }

    /**
     * LeetCode 整数反转
     * 时间复杂度为O(log10(x))，空间复杂度O(1)
     *
     * @param x
     * @return
     */
    public static int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int tmp = x % 10;
            // Integer最大值2^31-1 2147483647,Integer最小值-2^32 -2147483648
            // 如果rev大于Integer最大值/10，一定溢出；
            // 如果rev等于Integer最大值/10,并且最后一位tmp>7,从正数这边溢出；
            // 如果rev小于Integer最小值/10，一定溢出；
            // 如果rev等于Integer最小值/10,并且最后一位tmp<-8,从负数这边溢出；
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && tmp > 7))
                return 0;
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && tmp < -8))
                return 0;
            rev = rev * 10 + tmp;
            x = x / 10;
        }
        return rev;
    }

    // LeetCode 找到第一个只出现一次的字母
    public static int firstUniqChar(String s) {
        int[] num = new int[26];
        for (int i = 0; i < s.length(); i++) {
            num[s.charAt(i) - 'a'] += 1;
        }
        for (int i = 0; i < s.length(); i++) {
            if (num[s.charAt(i) - 'a'] == 1)
                return i;
        }
        return -1;
    }

    /**
     * LeetCode 1138 字母板上的路径
     *
     * @param target
     * @return
     */
    public static String alphabetBoardPath(String target) {
        int r = 0, c = 0;
        String ans = "";
        for (int i = 0; i < target.length(); i++) {
            int move_num_r = (target.charAt(i) - 'a') / 5 - r;
            int move_num_c = (target.charAt(i) - 'a') % 5 - c;
            char move_char_r = 'D', move_char_c = 'R';
            if (move_num_r < 0)
                move_char_r = 'U';
            if (move_num_c < 0)
                move_char_c = 'L';
            r += move_num_r;
            c += move_num_c;
            int move_r = Math.abs(move_num_r);
            int move_c = Math.abs(move_num_c);
            if (target.charAt(i) == 'z' && move_num_c != 0)
                move_r--;
            while (move_r-- > 0) {
                ans += move_char_r;
            }
            while (move_c-- > 0) {
                ans += move_char_c;
            }
            if (target.charAt(i) == 'z' && move_num_c != 0)
                ans += "D";
            ans += "!";
        }
        return ans;
    }

    /**
     * LeetCode 25 只出现一次的数字
     * 利用异或操作符，任何一个数和自身异或是0
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。
     * 找出那个只出现了一次的元素。
     * 用0初始化，依次和整个数组的元素依次异或，最后得到的就是只出现了一次的数字
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            ans = ans ^ nums[i];
        }
        return ans;
    }

    /**
     * LeetCode 26 两个数组的交集 II
     * 给定两个数组，编写一个函数来计算它们的交集。
     * 输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
     * 我们可以不考虑输出结果的顺序。
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null)
            return null;
        //先对两个数组进行排序
        sort(nums1);
        sort(nums2);
        int i1 = 0, i2 = 0;
        int i = 0;
        while (i1 < nums1.length && i2 < nums2.length) {
            if (nums1[i1] == nums2[i2]) {
                //发现相同的的元素，将其赋给nums1对应的位置
                nums1[i] = nums1[i1];
                i++;
                i1++;
                i2++;
            } else if (nums1[i1] < nums2[i2])
                i1++;
            else i2++;
        }
        int[] ans = new int[i];
        for (int j = 0; j < i; j++) {
            ans[j] = nums1[j];
        }
        return ans;
    }

    /**
     * LeetCode 35 有效的字母异位词
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if (s == null || t == null)
            return false;
        if (s.length() != t.length())
            return false;
        char[] char_s = s.toCharArray();
        char[] char_t = t.toCharArray();
        sort(char_s);
        sort(char_t);
        for (int i = 0; i < char_s.length; i++) {
            if (char_s[i] != char_t[i])
                return false;
        }
        return true;
    }

    /**
     * leetcode 27 加一
     *
     * @param digits
     * @return
     */
    public static int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0)
            return null;
        int len = digits.length;
        int[] tmp = new int[len + 1];
        int c = 1;
        int i;
        for (i = len - 1; i >= 0; i--) {
            if (digits[i] + c > 9) {
                c = 1;
                tmp[i + 1] = digits[i] = 0;
            } else {
                tmp[i + 1] = digits[i] = digits[i] + c;
                i--;
                break;
            }
        }
        // i == -1 表示多一位 返回tmp
        if (i == -1) {
            tmp[0] = 1;
            return tmp;
        } else return digits;
    }

    //LeetCode 28 移动零
    public void moveZeroes(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        int zeronum = 0;
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] == 0)
                zeronum++;
            else {
                nums[i - zeronum] = nums[i];
            }
        }
        while (zeronum > 0) {
            nums[len - zeronum] = 0;
            zeronum--;
        }
    }

    //LeetCode 29 两数之和
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> maps = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (maps.containsKey(target - nums[i])) {
                return new int[]{maps.get(target - nums[i]), i};
            } else {
                maps.put(nums[i], i);
            }
        }
        return null;
    }

    /**
     * LeetCode 30 有效的数独
     *
     * @param board
     * @return
     */
    public boolean isValidSudoku(char[][] board) {
        int size = 9;
        int[][] rows = new int[size][size + 1];
        int[][] cols = new int[size][size + 1];
        int[][] squares = new int[size][size + 1];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                int v = board[i][j] - '0';

                if (v < 1 || v > 9) continue;

                if (rows[i][v] > 0)  //第i行出现过当前数字
                    return false;
                if (cols[j][v] > 0)  //第j列出现过当前数字
                    return false;
                if (squares[i / 3 * 3 + j / 3][v] > 0)  //(i,j)所属的小方格出现过当前数字
                    return false;
                rows[i][v] = cols[j][v] = squares[i / 3 * 3 + j / 3][v] = 1;
            }
        return true;
    }

    /**
     * LeetCode 31 旋转图像
     * 顺时针旋转 等价于 先水平轴翻转，然后转置，记住了！！
     *
     * @param matrix
     */
    //先转置，再列交换
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //转置
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) continue;
                int tmp = matrix[j][i];
                matrix[j][i] = matrix[i][j];
                matrix[i][j] = tmp;
            }
        }
        //列交换
        for (int j = 0; j < n / 2; j++) {
            for (int i = 0; i < n; i++) {
                int tmp = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = matrix[i][j];
                matrix[i][j] = tmp;
            }
        }
    }

    // LeetCode 36 有效的回文子串
    public static boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        s = s.toLowerCase();
        while (i <= j) {
            char p = s.charAt(i);
            char q = s.charAt(j);
            if (!Character.isLetterOrDigit(p)) {
                i++;
                continue;
            }
            if (!Character.isLetterOrDigit(q)) {
                j--;
                continue;
            }
            if (p != q) return false;
            i++;
            j--;
        }
        return true;
    }

    /**
     * LeetCode 895 寻找两个有序数组的中位数
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        double mid = 0;
        int len1, len2;

        //nums1为空
        if (nums1 == null || nums1.length == 0) {
            len2 = nums2.length;
            mid = len2 % 2 == 0 ? (nums2[len2 / 2] + nums2[len2 / 2 - 1]) / 2.0 : nums2[len2 / 2];
            return mid;
        }

        //nums2为空
        if (nums2 == null || nums2.length == 0) {
            len1 = nums1.length;
            mid = len1 % 2 == 0 ? (nums1[len1 / 2] + nums1[len1 / 2 - 1]) / 2.0 : nums1[len1 / 2];
            return mid;
        }

        len1 = nums1.length;
        len2 = nums2.length;
        int i = 0, j = 0, count = 0, sumlen = len1 + len2;
        int[] nums = new int[sumlen];
        //两个数组从左边开始寻找中位数
        while (i < len1 || j < len2) {
            boolean use_nums2;
            //nums2遍历完
            if (j >= len2)
                use_nums2 = false;
                //nums1遍历完
            else if (i >= len1)
                use_nums2 = true;
                //二者均未遍历完，nums1小
            else if (nums1[i] <= nums2[j])
                use_nums2 = false;
                //二者均未遍历完，nums2小
            else
                use_nums2 = true;
            if (use_nums2)
                nums[count] = nums2[j++];
            else nums[count] = nums1[i++];

            //如果遍历完一半数组
            if (count == sumlen / 2) {
                mid = sumlen % 2 == 0 ? (nums[count] + nums[count - 1]) / 2.0 : nums[count];
                break;
            }
            count++;
        }
        return mid;
    }

    /**
     * LeetCode 22 买卖股票的最佳时机 II
     *
     * @param prices
     * @return
     */
//    public static int maxProfit(int[] prices) {
//        //采用贪心算法，如果第二天股票票价高于当天，就当天买入，第二天卖出
//        //否则就等第二天再买入
//        //采用贪心算法，如果第二天股票票价高于当天，就当天买入，第二天卖出
//        //否则就等第二天再买入
//        if (prices == null || prices.length <= 1)
//            return 0;
//        int profit = 0;
//        int tmp;
//
//        for (int i=1;i<prices.length;i++){
//            tmp = prices[i] - prices[i-1];
//            if (tmp>0)
//                profit += tmp;
//        }
//        return profit;
//    }

//LeetCode 买卖股票的最佳时机
//    以截至到上一天最低价格买入，当天价格卖出，即为前k天的最高利润
    public static int maxProfit(int[] prices) {
        if (prices.length <= 1) return 0;
        int maxprofit = 0;
        int days = prices.length;
        int[] profit = new int[days];
        int[] mincost = new int[days];
        profit[0] = 0;
        mincost[0] = prices[0];
        for (int i = 1; i < days; i++) {
            mincost[i] = mincost[i - 1] < prices[i - 1] ? mincost[i - 1] : prices[i - 1];
            profit[i] = prices[i] - mincost[i];
            if (profit[i] > maxprofit)
                maxprofit = profit[i];
        }
        return maxprofit;
    }

    //Leetcode 54 爬楼梯

    /**
     * 动态规划，一次可以爬一阶或者两阶
     * 爬n（n>=3)阶,看最后一步是爬了一个台阶，还是两个台阶
     */
    public int climbStairs(int n) {
        if (n <= 2)
            return n;
        int[] nums = new int[n + 1];
        nums[1] = 1;
        nums[2] = 2;
        for (int i = 3; i <= n; i++) {
            nums[i] = nums[i - 1] + nums[i - 2];
        }
        return nums[n];
    }

//    //leetcode 37 字符串转换整数（atoi）
//    public static int myAtoi(String str) {
//        if (str == null) return 0;
//        //去除空格
//        str = str.trim();
//        if (str.length() == 0) return 0;
//        char first = str.charAt(0);
//        //第一个非空字符不是数字或正负号
//        if (!Character.isDigit(first) && first != '+' && first != '-') return 0;
//        int num = 0;
//        int flag = 1;
//        if (first == '-') flag = -1;
//        else if (first != '+') num = first - '0';
//        for (int i = 1; i < str.length(); i++) {
//            char c = str.charAt(i);
//            if (!Character.isDigit(c)) break;
//            int tmp = c - '0';
//            if (flag > 0 && num > (Integer.MAX_VALUE - tmp) / 10)
//                return Integer.MAX_VALUE;
//            if (flag < 0 && num * flag < (Integer.MIN_VALUE - flag * tmp) / 10)
//                return Integer.MIN_VALUE;
//            num = num * 10 + tmp;
//        }
//        return num * flag;
//    }

    //LeetCode 38 实现strStr()
    public static int strStr(String haystack, String needle) {
        if (needle.length() == 0)
            return 0;
        if (haystack == null || haystack.length() == 0 || haystack.length() < needle.length())
            return -1;
        int hlen = haystack.length(), nlen = needle.length();
        for (int i = 0; i <= hlen - nlen; i++) {
            if (haystack.substring(i, i + nlen).equals(needle))
                return i;
        }
        return -1;
    }

    //Leetcode 39 报数
    public static String countAndSay(int n) {
        String result = "1";
        while (n-- > 1) {
            StringBuffer tmp = new StringBuffer();
            for (int i = 0; i < result.length(); i++) {
                char c = result.charAt(i);
                int count = 1;
                while (i + 1 < result.length() && c == result.charAt(i + 1)) {
                    i++;
                    count++;
                }
                tmp.append(count);
                tmp.append(c);
            }
            result = tmp.toString();
        }
        return result;
    }

    //Leetcode 40 最长公共前缀
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        if (strs.length == 1) return strs[0];
        int minlen = Integer.MAX_VALUE, min = 0;
        for (int i = 0; i < strs.length; i++) {
            if (strs[i].length() < minlen) {
                minlen = strs[i].length();
                min = i;
            }
        }
        String pref = "";
        String minstr = strs[min];
        for (int i = 0; i < minstr.length(); i++) {
            pref = minstr.substring(0, i + 1);
            for (String str : strs) {
                if (!str.startsWith(pref)) {
                    return pref.substring(0, pref.length() - 1);
                }
            }
        }
        return pref;
    }

    public static int findSmallNum(String word) {
        int[] ints = new int[word.length()];
        for (int j = 0; j < word.length(); j++) {
            ints[j] = word.charAt(j) - 'a';
        }
        sort(ints);
        int k = 0;
        while (k + 1 < ints.length && ints[k] == ints[k + 1]) k++;
        return k + 1;
    }

    // leetcode 5168 比较字符串最小字母出现频次
    public static int[] numSmallerByFrequency(String[] queries, String[] words) {
        int qlen = queries.length;
        int wlen = words.length;
        int[] query_ints = new int[qlen];
        int[] word_ints = new int[wlen];
        int[] answer = new int[qlen];
        for (int i = 0; i < wlen; i++) {
            word_ints[i] = findSmallNum(words[i]);
        }
        sort(word_ints);
        for (int i = 0; i < qlen; i++) {
            query_ints[i] = findSmallNum(queries[i]);
            int j = 0;
            while (j < wlen && word_ints[wlen - 1 - j] > query_ints[i]) j++;
            answer[i] = j;
        }
        return answer;
    }

    class Transaction {

    }

    // leetcode 5167 查询无效交易
    public static List<String> invalidTransactions(String[] transactions) {
        List<String> answer = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<Integer> times = new ArrayList<>();
        List<Integer> amounts = new ArrayList<>();
        List<String> cities = new ArrayList<>();
        int len = transactions.length;

        for (int i = 0; i < len; i++) {
            String[] strs = transactions[i].split(",");
            names.add(strs[0]);
            times.add(Integer.valueOf(strs[1]));
            amounts.add(Integer.valueOf(strs[2]));
            cities.add(strs[3]);
        }

        boolean[] invaild = new boolean[len];
        for (int i = 0; i < len; i++) {
            if (invaild[i]) continue;
            if (amounts.get(i) > 1000) {
                invaild[i] = true;
                continue;
            }
            for (int j = 0; j < len; j++) {
                if (j == i) continue;
                if (Math.abs(times.get(j) - times.get(i)) <= 60 && names.get(j).equals(names.get(i)) && !cities.get(j).equals(cities.get(i)))
                    invaild[j] = invaild[i] = true;
            }
        }

        for (int i = 0; i < len; i++) {
            if (invaild[i]) {
                StringBuffer buffer = new StringBuffer();
                buffer.append(names.get(i));
                buffer.append(",");
                buffer.append(times.get(i));
                buffer.append(",");
                buffer.append(amounts.get(i));
                buffer.append(cities.get(i));
                answer.add(buffer.toString());
            }
        }

        return answer;
    }


    //判断是否为质数
    public static boolean isPrime(int n) {
        if (n == 1) return false;
        if (n == 2) return true;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    //求全排列
    public static Long permutation(int n) {
        Long ans = 1L;
        for (int i = 2; i <= n; i++) {
            ans = ans * i;
            ans = ans % (int) (Math.pow(10, 9) + 7);
        }
        return ans;
    }

    // Leetcode 质数排列
    public static int numPrimeArrangements(int n) {
        int count = 0;
        for (int i = 2; i <= n; i++) {
            if (isPrime(i)) count++;
        }
        Long ans1 = permutation(count) % (int) (Math.pow(10, 9) + 7);
        long ans2 = permutation(n - count) % (int) (Math.pow(10, 9) + 7);
        return (int) (ans1 * ans2 % (int) (Math.pow(10, 9) + 7));
    }

    // LeetCode 健身计划评估
    public static int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
        int score = 0;
        for (int i = 0; i < calories.length - k + 1; i++) {
            int sum = 0;
            for (int j = 0; j < k; j++) {
                sum += calories[i + j];
            }
            if (sum < lower) score--;
            if (sum > upper) score++;
        }
        return score;
    }

    public static List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        List<Integer> nums = new ArrayList<>();
        int word_num = words.length;
        HashMap<Integer, HashSet<Integer>> word_chars = new HashMap<>();
        //标记word中出现的字母 用一个set保存
        for (int i = 0; i < word_num; i++) {
            HashSet<Integer> first = new HashSet<>();
            first.add(words[i].charAt(0) - 'a');
            word_chars.put(i, first);
            for (int j = 1; j < words[i].length(); j++) {
                char c = words[i].charAt(j);
                HashSet<Integer> set = word_chars.get(i);
                set.add(c - 'a');
                word_chars.replace(i, set);
            }
        }
        int puzzle_num = puzzles.length;
        int[][] puzzle_chars = new int[puzzle_num][26];
        //逐个遍历puzzle
        for (int i = 0; i < puzzles.length; i++) {
            //标记puzzle中出现的字母
            for (int j = 0; j < puzzles[i].length(); j++) {
                puzzle_chars[i][puzzles[i].charAt(j) - 'a'] = 1;
            }
            int count = 0;
            //逐个比对word
            for (int j = 0; j < word_num; j++) {
                //是否可以作为谜底的标志
                boolean flag = true;
                //如果当前word不包含puzzle的首字母，则不再检查word的字母
                HashSet<Integer> set = word_chars.get(j);
                if (!set.contains(puzzles[i].charAt(0) - 'a'))
                    continue;
                //获取word对应的set
                Object[] array = set.toArray();
                //对word中逐个单词进行查询
                for (int k = 0; k < array.length; k++) {
                    //未查询到word的当前字母
                    if (puzzle_chars[i][(int) array[k]] == 0) {
                        flag = false;
                        break;
                    }
                }
                //添加当前word
                if (flag) count++;
            }
            nums.add(count);
        }
        return nums;
    }

    //leetcode 合并两个有序数组
    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int i1 = m - 1;
        int i2 = n - 1;
        for (int i = m + n - 1; i >= 0; i--) {
            if (i1 == -1) nums1[i] = nums2[i2--];
            else if (i2 == -1) nums1[i] = nums1[i1--];
            else if (nums1[i1] > nums2[i2]) nums1[i] = nums1[i1--];
            else nums1[i] = nums2[i2--];
        }
    }

//    // leetcode 最长回文子串 动态规划求解
//    public static String longestPalindrome(String s) {
//        if (s.length() <= 1) return s;
//        int length = s.length();
//        int maxlength = 0;
//        String maxPalindrome = "";
//        boolean[][] dp = new boolean[length][length];
//        for (int j = 0 ; j < length; j++){
//            for (int i = j; i >= 0;i--){
//                char c1 = s.charAt(i), c2 = s.charAt(j);
//                dp[i][j] = true; //缺省设置为 i=j的情况
//                if (j <= i + 2) dp[i][j] = (c1 == c2);
//                else dp[i][j] = (c1 == c2) && dp[i+1][j-1];
//                if (dp[i][j] && j - i + 1 > maxlength) {
//                    maxPalindrome = s.substring(i, j + 1);
//                    maxlength = j - i + 1;
//                }
//            }
//        }
//        return maxPalindrome;
//    }

    // leetcode 最长回文子串 manacher算法
    public static String longestPalindrome(String s) {
        if (s.length() <= 1) return s;

        int length = s.length();
        int maxlength = 0;
        int start = 0, end = 0; //最长回文子串的起始长度

        //表示以当前字符为中心的回文子串的最右端到当前字符的长度
        int[] len = new int[2 * length + 1];

        //在s的每个字符之间都插上一个"#" 前后加上"#"
        String s_new = "#";
        for (int i = 0; i < length; i++) {
            s_new = s_new + s.charAt(i);
            s_new = s_new + "#";
        }

        // 之前匹配到的最长回文子串的右端点位置
        int p = 0;
        // 对应匹配到p的中心字符位置
        int po = 0;
        len[0] = 1;

        for (int i = 1; i < s_new.length(); i++) {
            // i没超过p的位置
            if (i <= p) {
                // 找到i关于po的对称位置
                int j = 2 * po - i;
                //如果以对称位置j为中心的回文串的一半长度（len[j]）小于 p和i之间的距离
                //等同于 len[j] <= p - i + 1
                if (len[j] < p - i) len[i] = len[j];
                else {  // 以i为中心的回文串可能会延伸到p之外
                    // 从p+1的位置开始匹配，中心po更新为i
                    // 注意限制下标范围
                    while (p + 1 < s_new.length() && 2 * i - p - 1 >= 0 && s_new.charAt(p + 1) == s_new.charAt(2 * i - p - 1))
                        p++;
                    po = i;
                    len[i] = p - i + 1;
                }
            } else {
                int con = 1;
                len[i] = 1;
                while (i + con < s_new.length() && i - con >= 0 && s_new.charAt(i + con) == s_new.charAt(i - con)) {
                    con++;
                    len[i]++;
                }
                if (i + con - 1 > p) {
                    // 注意这里p的赋值 因为con从1开始加，所以最后要减去1
                    p = i + con - 1;
                    po = i;
                }
            }
            //更新回文串最大长度
            if (len[i] > maxlength) {
                maxlength = len[i];
                //更新回文串的起始下标
                // 注意起始下标的赋值 求出来以i为中心的最长回文子串在s_new中的其实下标 然后除以2 就是 在s中的起始下标
                start = (i - (len[i] - 1)) / 2;
                end = start + len[i] - 1;
            }
        }
        return s.substring(start, end);
    }

    // leetcode 打家劫舍
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] robs = new int[nums.length];
        robs[0] = nums[0];
        robs[1] = nums[0] > nums[1] ? nums[0] : nums[1];
        for (int i = 2; i < nums.length; i++) {
            robs[i] = robs[i - 2] + nums[i] > robs[i - 1] ? robs[i - 2] + nums[i] : robs[i - 1];
        }
        return robs[nums.length - 1];
    }

    // leetcode 三数之和
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3) return res;
        sort(nums);
        for (int i = 0; i < nums.length - 2 && nums[i] <= 0; i++) {
            // 过滤连续重复的值  !!!重要！！！
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1, r = nums.length - 1;
            int sum = -nums[i];
            while (l < r) {
                // if (nums[r] + nums[r-1] < sum) break;
                if (nums[l] + nums[r] == sum) {
                    int[] a = {nums[i], nums[l], nums[r]};
                    List<Integer> tripe = stream(a).boxed().collect(Collectors.toList());
                    // if (!res.contains(tripe))
                    res.add(tripe);
                    // 去重
                    while (l < r && nums[l] == nums[l + 1]) l++;
                    while (l < r && nums[r] == nums[r - 1]) r--;
                    l++;
                    r--;
                } else if (nums[l] + nums[r] > sum)
                    r--;
                else l++;
            }
        }
        return res;
    }

    public static int countPrimes(int n) {
        if (n < 3) return 0;
        int count = 0;
        int[] not_prime = new int[n + 1];
        for (int i = 2; i < n; i++) {
            // 如果当前数字是已出现过质数的倍数，跳过检查
            if (not_prime[i] > 0) continue;
            // 从2开始，无需检查是不是质数，因为一定是质数不是任何质数的倍数）
            count++;
            for (int j = 2; i * j <= n; j++) {
                not_prime[i * j] = 1;
            }
        }
        return count;
    }

    // leetcode 有效的括号
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            if (c == '(' || c == '{' || c == '[')
                stack.push(c);
            else {
                if (stack.empty()) return false;
                Character top = stack.peek();
                switch (c) {
                    case ')':
                        if (top != '(') return false;
                        stack.pop();
                        continue;
                    case ']':
                        if (top != '[') return false;
                        stack.pop();
                        continue;
                    case '}':
                        if (top != '{') return false;
                    default:
                        stack.pop();
                }
            }
        }
        if (!stack.empty()) return false;
        return true;
    }

    // leetcode 3的幂 不用连续或递归实现
    // 思路：如果n是3的幂，用换底公式求出n以3为底的对数一定是一个整数
    public static boolean isPowerOfThree(int n) {
        if (n == 0) return false;
        double k = Math.log10(n) / Math.log10(3);
        if ((int) k == k)  // 判断是否是整数的方法
            return true;
        else return false;
    }

    // leetcode 罗马数字转整数
    public static int romanToInt(String s) {
        if (s.length() == 0) return 0;
        int i = 0;
        int num = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            switch (c) {
                case 'I':  // 1
                    if (i + 1 < s.length() && (s.charAt(i + 1) == 'V' || s.charAt(i + 1) == 'X')) {
                        if (s.charAt(i + 1) == 'V') num += 4;
                        else num += 9;
                        i++;
                    } else {
                        int count = 1;
                        while (i + 1 < s.length() && s.charAt(i + 1) == 'I') {
                            i++;
                            count++;
                        }
                        num += count;
                    }
                    break;
                case 'V':  // 5
                    num += 5;
                    break;
                case 'X':  // 10
                    if (i + 1 < s.length() && (s.charAt(i + 1) == 'L' || s.charAt(i + 1) == 'C')) {
                        if (s.charAt(i + 1) == 'L') num += 40;
                        else num += 90;
                        i++;
                    } else {
                        int count = 1;
                        while (i + 1 < s.length() && s.charAt(i + 1) == 'X') {
                            i++;
                            count++;
                        }
                        num += count * 10;
                    }
                    break;
                case 'L':  // 50
                    num += 50;
                    break;
                case 'C': // 100
                    if (i + 1 < s.length() && (s.charAt(i + 1) == 'D' || s.charAt(i + 1) == 'M')) {
                        if (s.charAt(i + 1) == 'D') num += 400;
                        else num += 900;
                        i++;
                    } else {
                        int count = 1;
                        while (i + 1 < s.length() && s.charAt(i + 1) == 'C') {
                            i++;
                            count++;
                        }
                        num += count * 100;
                    }
                    break;
                case 'D':  // 500
                    num += 500;
                    break;
                case 'M':  // 1000
                    int count = 1;
                    while (i + 1 < s.length() && s.charAt(i + 1) == 'M') {
                        i++;
                        count++;
                    }
                    num += count * 1000;
            }
            i++;
        }
        return num;
    }

    // leetcode 位1的个数
    public static int hammingWeight(int n) {
        String s = Integer.toBinaryString(n);
        System.out.println(s);
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '1') count++;
        }
        return count;
    }

    // leetcode 汉明距离
    public static int hammingDistance(int x, int y) {
        int n = x ^ y;
        return hammingWeight(n);
    }

    // leetcode 颠倒二进制位 未解决
    // 位操作相关基础
    // 两数交换：三次异或操作 a^=b b^=a a^=b
    // 变换符号：取反加1
    // -1的补码 0xffffffff 11111111111111111111111111111111
    // 对于任何数，与0异或都会保持不变，与-1即0xFFFFFFFF异或就相当于取反。
    // 对于负数可以通过对其取反后加1来得到正数。
    public static long reverseBits(int n) {
        return Integer.reverse(n);
    }

    // 求绝对值 不用任何判断表达式
    // 对n右移31位，相当于取符号位。如果n为正数，a等于0，n为负数，a等于-1
    // n为正数，a等于0，n和a异或等于n，再减去a还等于n
    // n为负数，a等于-1，n和a异或相当于对n取反，再减去a相当于+1，得到n的绝对值
    public static int my_abs(int n) {
        int a = n >> 31;
        return ((n ^ a) - a);
    }

    // leetcode 帕斯卡三角形 （杨辉三角）
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows <= 0) return res;

        List<Integer> row = new ArrayList<>();
        row.add(1);
        res.add(row);
        if (numRows == 1) return res;

        row = new ArrayList<>();
        row.add(1);
        row.add(1);
        res.add(row);
        if (numRows == 21) return res;

        for (int i = 3; i <= numRows; i++) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int j = 1; j < row.size(); j++) {
                tmp.add(row.get(j) + row.get(j - 1));
            }
            tmp.add(1);
            res.add(tmp);
            row = tmp;
        }
        return res;
    }

    // 采用异或解决 0^0=1^1=4^4=0 0^0^1^2^2^3 = 1
    // 数组下标是0到n-1，数组元素是0到n，缺一个数字，如果将结果初始化为n
    // 一直异或元素和下标，如果不缺数字，也就是0到n（下标到n-1加上初始化的n）异或0到n，最终等于0
    //所以最后得到的就是缺失的那个数字
    // 另一种方法就是用求和公式算出所有元素之和，然后减去实际的元素之和，就是缺失的数字
    public int missingNumber(int[] nums) {
        int res = nums.length;
        for (int i = 0; i < nums.length; i++) {
            res ^= nums[i] ^ i;
        }
        return res;
    }

    // leetcode 矩阵置零
    public void setZeroes(int[][] matrix) {
        if (matrix.length == 0) return;
        Set<Integer> zero_rows = new HashSet<>();
        Set<Integer> zero_cols = new HashSet<>();
        int row = matrix.length, col = matrix[0].length;

        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                if (matrix[i][j] == 0) {
                    zero_rows.add(i);
                    zero_cols.add(j);
                }

        Object[] rows = zero_rows.toArray();
        Object[] cols = zero_cols.toArray();

        for (int i = 0; i < rows.length; i++) {
            int crow = (int) rows[i];
            for (int j = 0; j < col; j++) {
                matrix[crow][j] = 0;
            }
        }

        for (int j = 0; j < cols.length; j++) {
            int ccol = (int) cols[j];
            for (int i = 0; i < row; i++) {
                matrix[i][ccol] = 0;
            }
        }
    }

    // leetcode 简化路径
    public static String simplifyPath(String path) {
        List<String> paths = new ArrayList<>();
        String[] strs = path.split("/");
        for (int i = 0; i < strs.length; i++) {
            String s = strs[i];
            System.out.println("spilt(): " + s);
            if (s.length() == 0) continue;
            if (s.equals(".")) continue;
            // 返回上一级
            if (s.equals("..")) {
                if (paths.size() > 0)
                    paths.remove(paths.size() - 1);
            } else
                paths.add(s);
        }
        String res = "";
        for (int i = 0; i < paths.size(); i++) {
            res += "/" + paths.get(i);
        }
        // 最终返回根目录
        if (paths.size() == 0) res = "/";
        return res;
    }

    // leetcode 搜索旋转排序数组 思路很关键!!!
    // 关键思路：mid将数组分为左右两半，肯定一半是有序的，一半跨过左右各半段有序数组
    // 只需确定target可能在数组的哪一半即可，每次都用有序数组那一半判断
    public int search(int[] nums, int target) {
        if (nums.length == 0)
            return -1;
        int l = 0, r = nums.length - 1;
        while (l <= r && l >= 0 && l < nums.length && r >= 0 && r < nums.length) {
            int mid = (l + r) / 2;
            if (target == nums[mid])
                return mid;
            // 比如 3 大于nums[6] = 2, 小于nums[0] = 7 因为是只旋转了一次的有序数组，所以不可能存在
            if (target > nums[r] && target < nums[l])
                return -1;
                // 左半段数组是有序的
            else if (nums[l] <= nums[mid]) {
                //target在里边
                if (nums[l] <= target && target <= nums[mid])
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            //左半段数组是左大右小型的
            else {
                // 此时右半段数组是有序的，只需判断右半段
                if (nums[mid] < target && target <= nums[r])
                    l = mid + 1;
                else r = mid - 1;
            }
        }
        return -1;
    }

    // leetcode 复原IP地址
    static List<String> result;
    static int[] digits;

    /**
     * 回溯，为了加快速度可以提前计算下是否可行
     * part 计算的是四个IP地址段中的哪一段
     * value 当前IP地址段的值
     * ip 当前得到的IP
     * index digit数组下一个要访问的索引
     */
    public static void find(int part, int value, String ip, int index) {
        // 如果搜索段数超过3，返回
        // 如果搜索到digit末尾，返回
        if (part > 3 || index == digits.length) {
            if (part == 3 && index == digits.length)  // 如果part等于3并且正好到digits结尾，添加合法IP
                result.add(ip);
            return;
        }

        System.out.println(ip);
        int tmp = value * 10 + digits[index];
        // 如果当前IP段还小于255，继续向后搜索
        // 注意 这里如果当前IP段第一个数字为0 也即此时value等于0，不合法，停止搜索
        if (tmp <= 255 && value != 0) {
            find(part, tmp, ip + digits[index], index + 1);
        }
        // 搜索下一IP段
        find(part + 1, digits[index], ip + "." + digits[index], index + 1);
    }

    // leetcode 复原IP地址
    public static List<String> restoreIpAddresses(String s) {
        result = new ArrayList<>();
        if (s.length() < 4 || s.length() > 32)
            return result;
        digits = new int[s.length()];
        for (int i = 0; i < s.length(); i++)
            digits[i] = s.charAt(i) - '0';
        find(0, digits[0], "" + digits[0], 1);
        return result;
    }

    // 46. 全排列
    public void permuteBackTrack(List<List<Integer>> permutations, List<Integer> permutation, int[] nums, boolean[] exist, int k) {
        if (k == nums.length) {
            permutations.add(new ArrayList<>(permutation));
            return;
        }
        for (int j = 0; j < nums.length; j++) {
            if (!exist[j]) {
                exist[j] = true;
                permutation.add(nums[j]);
                permuteBackTrack(permutations, permutation, nums, exist, k + 1);
                permutation.remove(permutation.size() - 1);
                exist[j] = false;
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0)
            return res;
        List<List<Integer>> permutations = new ArrayList<>();
        List<Integer> permutation = new ArrayList<>();
        boolean[] exist = new boolean[nums.length];
        permuteBackTrack(permutations, permutation, nums, exist, 0);
        return permutations;
    }

    // int[] List<Integer> Integer[] 相互转换
    public void inttolist() {
        List<Integer> list = new ArrayList<>();
        list.add(111);
        // List<Integer> -> int[]
        int[] ints = list.stream().mapToInt(Integer::valueOf).toArray();
        System.out.println(ints.length);

        // List<Integer> -> Integer[]
        Integer[] integers1 = list.toArray(new Integer[0]);

        // int[] -> List<Integer>
        List<Integer> collect = stream(ints).boxed().collect(Collectors.toList());
        System.out.println(collect.size());

        // int[] -> Integer[]
        Integer[] integers = stream(ints).boxed().toArray(Integer[]::new);

        //Integer[] -> int[]
        int[] ints1 = stream(integers1).mapToInt(Integer::valueOf).toArray();

        //Integer[] -> list<Integer>
        List<Integer> integers2 = new ArrayList<>(asList(integers1));
    }

    // leetcode 字母异位词分组
    // 首先计算每个string中26个字母出现的次数，然后将其转换为string，以此作为k，对应的string加到value中
    // 最后将hashmap转化为list
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> hashMap = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            StringBuffer stringBuffer = new StringBuffer();
            int[] count = new int[26];
            for (int j = 0; j < strs[i].length(); j++)
                count[strs[i].charAt(j) - 'a']++;
            for (int k = 0; k < count.length; k++)
                stringBuffer.append(count[k] + '/');
            String key = stringBuffer.toString();
            // 如果不存在当前的key，则put
            if (!hashMap.containsKey(key)) {
                hashMap.put(key, new ArrayList<>());
            }
            hashMap.get(key).add(strs[i]);
        }
        return new ArrayList<>(hashMap.values());
    }

    // 递增的三元子序列
//    如果存在这样的 i, j, k,  且满足 0 ≤ i < j < k ≤ n-1，
//    使得 arr[i] < arr[j] < arr[k] ，返回 true ; 否则返回 false 。
//    注意初始化，这里使用int最大值初始化
//    维持一个 first second的元祖，记录前面出现的最小的两个数，first最小，second第二小
    public static boolean increasingTriplet(int[] nums) {
        if (nums.length < 3)
            return false;
        int first = Integer.MAX_VALUE, second = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < first)
                first = nums[i];
            else if (first < nums[i] && nums[i] < second)
                second = nums[i];
            else if (nums[i] > second)
                return true;
        }
        return false;
    }

    //     leetcode 岛屿数量
//     广度优先搜索BFS
//    找到一个1，就将其上下左右周围的元素都置为2，每次最先找到一个1，岛屿数目加1
//     https://blog.csdn.net/hi_baymax/article/details/82585480
//    public void searchneighbor(char[][] grid, int i, int j){
//        if (i < 0 || i == grid.length || j < 0 || j == grid[0].length || grid[i][j] != '1')
//            return;
//        grid[i][j] = '2';
//        searchneighbor(grid, i+1, j);
//        searchneighbor(grid, i-1, j);
//        searchneighbor(grid, i, j+1);
//        searchneighbor(grid, i, j-1);
//    }
//
//    public int numIslands(char[][] grid) {
//        int sum = 0;
//        for (int i = 0; i < grid.length; i++){
//            for (int j = 0; j < grid[0].length; j++){
//                if (grid[i][j] == '1'){
//                    sum++;
//                    searchneighbor(grid, i, j);
//                }
//            }
//        }
//        return sum;
//    }

    //     leetcode 岛屿数量
//    并查集解决
//    将数组以m*n形式 创建一个father数组，没找到一个1，就将其相邻的元素father置为1，最后f[i]=i并且grid对应位置为'1'的即为岛屿数量
    public static int findFather(int[] f, int a) {
        if (a == f[a])
            return a;
//        找到最终指向的f
        int last = a;
        while (f[last] != last) {
            last = f[last];
        }
//        让从a到last都指向last
        while (f[a] != a) {
            int tmp = f[a];
            f[a] = last;
            a = tmp;
        }
        return a;
    }

    public static void unionFather(int[] f, int t1, int t2) {
        int f1 = findFather(f, t1);
        int f2 = findFather(f, t2);
        if (f1 != f2) {
            f[f1] = f2;
        }
    }

    public static int numIslands(char[][] grid) {
        if (grid.length == 0)
            return 0;
        int sum = 0;
        int m = grid.length, n = grid[0].length;
        int[] f = new int[m * n];

        for (int i = 0; i < m * n; i++)
            f[i] = i;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    int t = i * n + j;
                    int t1 = t + 1, t2 = t + n;
                    if (j + 1 < n && grid[i][j] == grid[i][j + 1]) unionFather(f, t, t1);
                    if (i + 1 < m && grid[i][j] == grid[i + 1][j]) unionFather(f, t, t2);
                }
            }
        }
        for (int i = 0; i < m * n; i++) {
            if (f[i] == i && grid[i / n][i % n] == '1')
                sum++;
        }
        return sum;
    }

    // leetcode 子集	给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
    // 说明：解集不能包含重复的子集。
    // 用回溯算法解决，记得添加当前下标元素回溯结束后，要移除result的最后一个元素
    public List<List<Integer>> results = new ArrayList<>();

    public void backtrack(int[] nums, List<Integer> result, int k) {
        if (k == nums.length) {
            results.add(new ArrayList(result));
            return;
        }
        backtrack(nums, result, k + 1);
        result.add(nums[k]);
        backtrack(nums, result, k + 1);
        result.remove(result.size() - 1);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<Integer> result = new ArrayList<>();
        backtrack(nums, result, 0);
        return results;
    }

    //    leetcode 电话号码的字母组合
    List<String> res = new ArrayList<>();

    public void trackback(Map<Integer, String> digit2letters, String digits, String s, int k) {
        if (k == digits.length()) {
            res.add(s);
            return;
        }
        String letters = digit2letters.get(digits.charAt(k) - '0');
        for (int j = 0; j < letters.length(); j++) {
            trackback(digit2letters, digits, s + letters.charAt(j), k + 1);
        }
    }

    public List<String> letterCombinations(String digits) {

//        注意！此时没有任何组合 应该输出[]
        if (digits.length() == 0)
            return res;

        Map<Integer, String> digit2letters = new HashMap<>();
        digit2letters.put(1, "");
        digit2letters.put(2, "abc");
        digit2letters.put(3, "def");
        digit2letters.put(4, "ghi");
        digit2letters.put(5, "jkl");
        digit2letters.put(6, "mno");
        digit2letters.put(7, "pqrs");
        digit2letters.put(8, "tuv");
        digit2letters.put(9, "wxyz");

        String s = "";
        trackback(digit2letters, digits, s, 0);
        return res;
    }

    //     leetcode 单词搜索
//     回溯函数，k 当前搜索的char，direction，used 标记是否用过，row col分别为当前搜索的行和列
    public boolean searchWord(char[][] board, String word, int k, boolean[][] used, int row, int col) {
//         搜索到word结尾，表示成功搜索到word，返回true
        if (k == word.length())
            return true;
        if (col < 0 || col >= board[0].length || row < 0 || row >= board.length ||
                used[row][col] || board[row][col] != word.charAt(k)) {
            return false;
        }

        // System.out.println("k=" + k + ", row=" + row + ", col=" + col);
        used[row][col] = true;
        if (searchWord(board, word, k + 1, used, row, col - 1)
                || searchWord(board, word, k + 1, used, row, col + 1)
                || searchWord(board, word, k + 1, used, row - 1, col)
                || searchWord(board, word, k + 1, used, row + 1, col))
            return true;
        used[row][col] = false;
        return false;
    }

    public boolean exist(char[][] board, String word) {
        boolean flag = false;
        boolean[][] used = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == word.charAt(0)) {
                    int row = i, col = j;
                    // System.out.println("row="+row+", col="+col);
                    flag = searchWord(board, word, 0, used, row, col);
                    if (flag)
                        return flag;
                }
            }
        }
        return flag;
    }

    // leetcode 颜色分类
    //遍历两遍数组
//    public void sortColors(int[] nums) {
//        int[] numsofcolor = new int[3];
//        for (int i = 0; i < nums.length; i++){
//            if (nums[i] == 0) numsofcolor[0]++;
//            if (nums[i] == 1) numsofcolor[1]++;
//            if (nums[i] == 2) numsofcolor[2]++;
//        }
//
//        int k = 0;
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < numsofcolor[i]; j++) {
//                nums[k] = i;
//                k++;
//            }
//        }
//    }

//    遍历一次，常数空间

    public void swap(int[] a, int i, int j){
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    /**
     * 按照0 1 2排序，如果是0，和左边进行交换，如果是2，和右边进行交换，1的话则往右走
     * @param nums
     */
    public void sortColors(int[] nums) {
        int l = -1;
        int r = nums.length;
        int i = 0;
        while(i < r){
            if (nums[i] == 0){
                l++;
                // 将0交换到最左边
                swap(nums, i, l);
                // System.out.println("l: " + l + ", i: "+ i);
                i++;
            } else if (nums[i] == 1)
                i++;
            else if (nums[i] == 2){
                r--;
                // 将2交换到最右边
                swap(nums, i, r);
                // System.out.println("r: " + r + ", i: " + i);
            }
        }
    }

    // leetcode (中级算法-排序和搜索) 前k个元素
    // 解法一：排序算法
    // 时间复杂度 O(nlogn) 空间复杂度O(n)
//    public List<Integer> topKFrequent(int[] nums, int k) {
//
//        // 统计元素的频率
//        HashMap<Integer, Integer> freqMap = new HashMap<>();
//        for (Integer num : nums){
//            freqMap.put(num, freqMap.getOrDefault(num, 0) + 1);
//        }
//
//        //对元素按照频率进行降序排序
//        // 将map转换为list，使用Collections.sort函数
//        List<Map.Entry<Integer, Integer>> list = new ArrayList<>(freqMap.entrySet());
//        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
//            @Override
//            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
//                return o2.getValue() - o1.getValue();
//            }
//        });
//
//        // 取出前k个元素
//        List<Integer> res = new ArrayList<>();
//        int count = 0;
//        for (Map.Entry<Integer, Integer> entry : list){
//            res.add(entry.getKey());
//            if (++count == k)
//                break;
//        }
//        return res;
//    }

    // 解法二：堆排序 时间复杂度O(NlogK)
    // 维持一个大小为k保存前k个元素的最小堆
    // 如果当前元素的频率大于堆顶元素的频率，则删除堆顶元素，插入当前元素
    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> freqmap = new HashMap<>();
        // 统计元素出现的频率
        for (Integer num : nums){
            freqmap.put(num, freqmap.getOrDefault(num, 0) + 1);
        }

        // 构造前k个最大频率的最小堆（升序排列）
        PriorityQueue<Integer> pq = new PriorityQueue<>(
                new Comparator<Integer>() {
                    @Override
                    public int compare(Integer o1, Integer o2) {
                        return freqmap.get(o1) - freqmap.get(o2);
                    }
                }
        );
        for (Integer key: freqmap.keySet()){
            if (pq.size() < k){
                pq.add(key);
            }else if (freqmap.get(key) > freqmap.get(pq.peek())){
                pq.remove();
                pq.add(key);
            }
        }

        // 取出最小堆中元素
        List<Integer> res = new ArrayList<>();
        while(!pq.isEmpty()){
            res.add(pq.remove());
        }
        return res;
    }


    // leetcode 数组中的第k个最大元素

//     // 冒泡排序
//    public int findKthLargest(int[] nums, int k) {
//        int len = nums.length;
//        for (int i = 0; i < k; i++){
//            int max_i = 0;
//            for (int j = 0; j < len - i; j++){
//                if (nums[j] > nums[max_i])
//                    max_i = j;
//            }
//            // System.out.println(nums[max_i]);
//            int tmp = nums[len-1 - i];
//            nums[len-1 - i] = nums[max_i];
//            nums[max_i] = tmp;
//        }
//        return nums[len - k];
//    }
//
//    // 最小堆
//    public int findKthLargest(int[] nums, int k) {
//        PriorityQueue<Integer> pq = new PriorityQueue<>();
//        for (int i = 0; i < nums.length; i++){
//            if (i < k)
//                pq.add(nums[i]);
//            else if (nums[i] >= pq.peek()){
//                pq.remove();
//                pq.add(nums[i]);
//            }
//        }
//        return pq.peek();
//    }

    // java自带库函数
    public int findKthLargest(int[] nums, int k) {
        sort(nums);
        return nums[nums.length - k];
    }

    // leetcode 寻找峰值
    // 二分查找
    public int findPeakElement(int[] nums) {
        if (nums.length == 1)
            return 0;
        if (nums.length == 2)
            return nums[0] > nums[1] ? 0 : 1;
        int l = 0, r = nums.length-1;
        while(l <= r){
            int mid = (l + r) / 2;
            // System.out.println(mid);
            if ( (mid == 0 || mid > 0 && nums[mid] > nums[mid-1]) && (mid == nums.length-1 || nums[mid] > nums[mid+1]) )
                return mid;
            if (mid > 0 && nums[mid] < nums[mid-1])
                r = mid == 0 ? 0 : mid - 1;
            else l = mid == nums.length - 1 ? mid : mid + 1;
        }
        return l;
    }

    // leetcode 在排序数组中查找元素的第一个和最后一个位置
    // 二分查找
    public int[] searchRange(int[] nums, int target) {
        int l = 0, r = nums.length-1;
        int[] res = new int[2];
        res[0] = res[1] = -1;
        if (nums.length == 0)
            return res;
        while(l <= r){
            int mid = (l + r) / 2;
            if (nums[mid] == target){
                res[0] = res[1] = mid;
                while(mid > 0 && nums[--mid] == target)res[0] = mid;
                mid = (l + r) / 2;
                while(mid < nums.length-1 && nums[++mid] == target)res[1] = mid;
                return res;
            }
            if (nums[mid] > target)
                r = mid - 1;
            else l = mid + 1;
        }
        return res;
    }

    // leetcode 搜索二维矩阵II
    // 其实肯定是从一个角上开始搜索，必须是左上角吗？左上角元素是最小的，下边和右边都是更大的元素，右下角同理是最大的元素。
    // 所以选定从右上角（也可以是左下角）开始搜索，这样左边是更小的元素，下边是更大的元素，两边大小不一样，可以进行搜索。
    // 从矩阵右上角开始搜索，比target大了就向左走，比target小了就向右走，直到找到或者越过边界
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0)
            return false;
        int r = 0;
        int c = matrix[0].length - 1;
        while (r < matrix.length && c >= 0) {
            if (matrix[r][c] == target)
                return true;
            if (matrix[r][c] > target)
                c--;
            else r++;
        }
        return false;
    }

    // leetcode 跳跃游戏
    // 用一个int整型记录能到达的最后位置，不断更新
    public boolean canJump(int[] nums) {
        if (nums.length < 2)
            return true;
        int nextmax = nums[0];
        for (int i = 0; i < nums.length; i++) {
            nextmax = Math.max(nextmax, i + nums[i]);
            if (nextmax >= nums.length - 1)
                return true;
            // nums[i] 不是正数 表示不能再往后走 并且 最多能走到当前位置
            if (nums[i] <= 0 && nextmax <= i)
                return false;
        }
        return false;
    }

    // leetcode 零钱兑换 动态规划解法
    // 兑换n元零钱的方法：兑换n-1元零钱+1元硬币，兑换n-2元零钱+2元硬币，...，兑换1元零钱+n-1元硬币，兑换0元零钱+n元硬币
    public int coinChange(int[] coins, int amount) {
        if (amount == 0)
            return 0;
        if (coins.length == 0 || amount < 0)
            return -1;

        sort(coins);
        //兑换零钱所需硬币数量的数组
        int[] dp = new int[amount + 1];
        dp[0] = 0;
        dp[1] = coins[0] == 1 ? 1 : -1;
        //从兑换2元硬币，到兑换n元，动态规划
        // 兑换i元硬币的情况下
        for (int i = 2; i <= amount; i++) {
            // 从小到大遍历一遍硬币
            dp[i] = -1;
            // 记录暂时的最小值
            int tmp = amount;
            for (int j = 0; j < coins.length; j++) {
                // 如果当前硬币可用 并且 可以兑换i-coins[j]的零钱
                if (coins[j] <= i && dp[i - coins[j]] != -1) {
                    // 如果找到了至少一种有效的兑换方式，更新dp值
                    if (dp[i - coins[j]] < tmp) {
                        tmp = dp[i - coins[j]];
                        dp[i] = tmp + 1;
                    }
                }
            }
            // System.out.println("dp["+i+"]:" + dp[i]);
        }
        return dp[amount];
    }

    // leetcode 最长上升子序列
    // 动态规划+二分查找
    public int lengthOfLIS(int[] nums) {
        if (nums.length <= 1)
            return nums.length;
        // 维持一个最长的上升子序列
        List<Integer> maxseq = new ArrayList<>();
        maxseq.add(nums[0]);
        for (int i = 1; i < nums.length; i++) {
            // 当前值比最大元素大，直接加进去
            if (maxseq.get(maxseq.size() - 1) < nums[i])
                maxseq.add(nums[i]);
            else {
                // 否则的话，二分查找maxseq，找到第一个比当前大的最小的元素，然后进行替换
                int l = 0, r = maxseq.size() - 1;
                int mid = (l + r) / 2;
                while (l < r) {
                    mid = (l + r) / 2;
                    // if (maxseq.get(mid) > nums[i] && (mid == 0 || maxseq.get(mid-1) < nums[i]))
                    //     break;
                    // 这里不能有等号，严格的上升子序列
                    if (maxseq.get(mid) < nums[i])
                        l = mid + 1;
                    else r = mid;
                }
                // 这个地方要设置成对l位置赋值，不然会报错，为什么呢，你看最终l是mid右边的那个，是大于nums[i]的
                maxseq.set(l, nums[i]);
            }
        }
        return maxseq.size();
    }

    // leetcode Excel表列序号
    // "AB" 直接换算成26进制
    public int titleToNumber(String s) {
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            res = res * 26 + s.charAt(i) - 'A' + 1;
        }
        return res;
    }

    // leetcode Pow(x, n)
    // 二分思想递归，复杂度O(log2n)
    public double myPow(double x, int n) {
        if (n == -1) return 1. / x;
        if (n == 1) return x;
        if (n == 0) return 1;
        double half = myPow(x, n / 2);
        double rest = myPow(x, n % 2);
        return half * half * rest;
    }

    // leetcode 29 两数相除
    // 要求不能用乘法 除法和 mod 运算符。
    // 被除数和除数全部转换成负数，用负数做减法，大于号变小于，
    // 避免了-2147483648不能取绝对值的问题
    public int divide(int dividend, int divisor) {
        int sign = 1;
        int res = -1;
        int n = -1;
        if (dividend > 0) {
            sign = opposite(sign);
            dividend = opposite(dividend);
        }
        if (divisor > 0) {
            sign = opposite(sign);
            divisor = opposite(divisor);
        }

        // 被除数绝对值小于除数，负数情况下为被除数大于除数，直接返回0
        if (dividend > divisor) return 0;

        int div = divisor;
        // 先减一次，被除数和除数可能相等,res初始化为-1
        dividend -= divisor;

        // 这里等于也可以的哦！！不带等号的话答案会少一个1或者-1
        while (dividend <= div) {
            divisor += divisor;
            n += n;
            if (dividend > divisor) {
                divisor = div;
                n = -1;
            }
            dividend -= divisor;

            // 检查溢出边界，注意此时的res和n都是负数
            if (sign < 0 && res < Integer.MIN_VALUE - n)
                return Integer.MAX_VALUE;
            if (sign > 0 && opposite(res) > Integer.MAX_VALUE - opposite(n))
                return Integer.MAX_VALUE;
            res += n;
            // System.out.println("dividend=" + dividend);
            // System.out.println("divisor=" + divisor);
            // System.out.println("res=" + res);

        }

        return sign > 0 ? opposite(res) : res;
    }

    // 求相反数 按位取反+1
    public int opposite(int n) {
        return ~n + 1;
    }

    // leetcode 9 回文数
//    public boolean isPalindrome(int x) {
//        if (x < 0)
//            return false;
//        if (x >= 0 && x < 10)
//            return true;
//        if (x % 10 == 0)
//            return false;
//        int l, r;
//        // 求x的位数，n+1位
//        // java中Math.log求的是ln，以自然对数e为底,求其他底数的对数值，需除以Math.log(自己的底数)
//        int n = (int)(Math.log(x) / Math.log(10));
//        while(n>0){
//            // System.out.println(n);
//            l = x / (int)Math.pow(10, n) % 10;
//            r = x % 10;
//            // System.out.println("l=" + l + ",r=" + r);
//            if (l != r)
//                return false;
//            // 去掉末位，首位不能去，因为可能是0
//            x = x/10;
//            // System.out.println(x);
//            //求最高位，往低位走每次减一，然后x除了10，再减1
//            n -= 2;
//        }
//        return true;
//    }

    // leetcode 9 回文数
    // 翻转一半数字，当x小于reversenum时，翻转了一半的数字
    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0))
            return false;
        int reversenum = 0;
        while (x > reversenum) {
            int l = x % 10;
            reversenum = reversenum * 10 + l;
            x /= 10;
        }
        // 如果x位数为奇数，比如12321，最后x=12,reversenum=123
        // 那么reversenum的末位数就是中间的数，不影响回文，直接舍掉
        return x == reversenum || x == reversenum / 10;
    }

    // leetcode 27 移除元素
    public int removeElement(int[] nums, int val) {
        int n = nums.length, i = 0, k = nums.length - 1;
        // 从头遍历元素，遇到val值，从后往前找非val替换，同时更新n值
        while (i < n) {
            if (nums[i] == val) {
                n--;
                while (nums[k] == val) {
                    k--;
                    n--;
                }
                nums[i] = nums[k--];
            }
            i++;
        }
        // for (int j = 0; j < nums.length; j++)
        //     System.out.print(nums[j]+" ");
        return n;
    }

    // leetcode 69 x的平方根，二分法求解
    public int mySqrt(int x) {
        if (x <= 1) return x;
        // 端点都用long类型，中间计算mid用int会越界
        long l = 1, r = x / 2;
        while (l < r) {
            // 注意这里取有中位数，否则可能陷入死循环
            long mid = (l + r + 1) / 2;
            // System.out.println("mid="+mid+",left="+l+",right="+r+"\nmid*mid="+mid*mid+"\n");
            // 最后返回的是左端点，所以l = mid
            if (x == mid * mid) return (int) mid;
            if (x < mid * mid) r = mid - 1;
            else l = mid;

        }
        return (int) l;
    }

    // leetcode 快乐数
    public boolean isHappy(int n) {
        // n是10的指数
        if (n - Math.log(n) / Math.log(10) == 0)
            return true;
        HashSet<Integer> set = new HashSet();
        while (true) {
            int sum = 0;
            int tmp = n;
            while (tmp > 0) {
                // 计算各位平方和
                sum += (tmp % 10) * (tmp % 10);
                tmp /= 10;
            }
            if (sum == 1) return true;
            // 无限循环
            if (set.contains(sum)) return false;
            set.add(sum);
            n = sum;
        }
    }

//    // n能被几个a整除
//    public int logM(long n, int a){
//        int k = 1;
//        long s = a;  // s居然会越界，换成long
//        int num = 0;
//        while(n > 0 && n % a == 0){
//            if (n % s > 0){
//                k = 1;s = a;
//            }
//            num += k;
//            n /= s;
//            // 每次乘方，加快计算速度
//            k += k; s *= s;
//        }
//        return num;
//    }
//
//    // leetcode 阶乘后的0
//    public int trailingZeroes(int n) {
//        if (n < 5) return 0;
//        int num = 0;
//        long fact = 1;
//        for (; n >= 2; n--){
//
//            if (n % 5 != 0 && n % 2 != 0) continue;
//
//            int tmp = n;
//            int a = tmp;
//
//            // 对tmp取掉末尾的0
//            a = logM(tmp, 10);
//            num += a;
//            tmp /= (int)Math.pow(10, a);
//
//            // 对tmp取2
//            if (tmp % 2 == 0){
//                a = logM(tmp, 2);
//                tmp = (int)Math.pow(2, a);
//                fact *= tmp;
//            }
//            // 对tmp取5
//            if (tmp % 5 == 0){
//                a = logM(tmp, 5);
//                tmp = (int)Math.pow(5, a);
//                fact *= tmp;
//            }
//
//            if (n <= 21)System.out.println("tmp="+tmp+",fact="+fact+",num="+num);
//
//            // 每一步对fact取末尾的0，防止溢出
//            a = logM(fact, 10);
//            num += a;
//            fact /= (int)Math.pow(10, a);
//            if (n <= 21)System.out.println("fact="+fact);
//            // 对fact取2
//            if (fact % 2 == 0){
//                a = logM(fact, 2);
//                fact = (int)Math.pow(2, a);
//            }
//            // 对fact取5
//            if (fact % 5 == 0){
//                a = logM(fact, 5);
//                fact = (int)Math.pow(5, a);
//            }
////            if (n <= 22)
////                System.out.println("fact="+fact+",n="+n+",num="+num+"\n");
//        }
//        return num;
//    }

    // leetcode 阶乘后的零，统计n!的质因数5的个数
    public int trailingZeroes(int n) {
        long num = 0;
        while (n > 0) {
            num += (n / 5);
            n /= 5;
        }
        return (int) num;
    }


    // leetcode 166.分数到小数
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        long num = numerator;
        long den = denominator;
        String sign = "";
        //确定符号
        if (num > 0 && den < 0 || num < 0 && den > 0) {
            sign = "-";
        }
        //转为正数
        num = Math.abs(num);
        den = Math.abs(den);
        //记录整数部分
        long integer = num / den;
        //计算余数
        num = num - integer * den;
        HashMap<Long, Integer> map = new HashMap<>();
        int index = 0;
        String decimal = "";//记录小数部分
        int repeatIndex = -1;//保存重复的位置
        while (num != 0) {
            num *= 10;//余数乘以 10 作为新的被除数
            if (map.containsKey(num)) {
                repeatIndex = map.get(num);
                break;
            }
            //保存被除数
            map.put(num, index);
            //保存当前的商
            long decimalPlace = num / den;
            //加到所有的商中
            decimal = decimal + decimalPlace;
            //计算新的余数
            num = num - decimalPlace * den;
            index++;
        }
        //是否存在循环小数
        if (repeatIndex != -1) {
            String dec = decimal;
            return sign + integer + "." + dec.substring(0, repeatIndex) + "(" + dec.substring(repeatIndex) + ")";
        } else {
            if (decimal == "") {
                return sign + integer;
            } else {
                return sign + integer + "." + decimal;
            }
        }
    }

    // leetcode 两整数之和
    // 三步循环：1.计算无进位加法，2.计算加法进位，3.将一二步结果相加，方法还是1.2.，也即对得到的结果重复一二步操作，直至进位为0。
    // 用位运算实现，计算无进位加法，即两数异或；计算加法进位，即两数相与。
    public int getSum(int a, int b) {
        while (b != 0) {
            int tmp = a & b;
            a = a ^ b; // 计算无进位加法
            b = tmp << 1;  // 计算加法进位
        }
        return a;
    }

    // 逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Stack<Integer> s = new Stack<>();
        for (String token : tokens) {
            // 判断字符串是否是整数，包括正负，？0或1个，+ 1或多个， * 0或多个
            if (token.matches("-?[0-9]+")) {
                s.push(Integer.parseInt(token));
                continue;
            }
            int tmp1, tmp2;
            if ("+".equals(token)) {
                tmp1 = s.pop();
                tmp2 = s.pop();
                s.push(tmp1 + tmp2);
            } else if ("-".equals(token)) {
                tmp2 = s.pop();
                tmp1 = s.pop();
                s.push(tmp1 - tmp2);
            } else if ("*".equals(token)) {
                tmp1 = s.pop();
                tmp2 = s.pop();
                s.push(tmp1 * tmp2);
            } else if ("/".equals(token)) {
                tmp2 = s.pop();
                tmp1 = s.pop();
                s.push(tmp1 / tmp2);
            }
        }
        return s.peek();
    }

//    // leetcode 逆波兰表达式求值，节约时间的版本，不用判断是否是整数
//    public int evalRPN(String[] tokens) {
//        Stack<Integer> s = new Stack<>();
//        for (String token: tokens){
//            int tmp1, tmp2;
//            if ("+".equals(token)){
//                tmp1 = s.pop();tmp2 = s.pop();
//                s.push(tmp1+tmp2);
//            }else if ("-".equals(token)){
//                tmp2 = s.pop(); tmp1 = s.pop();
//                s.push(tmp1-tmp2);
//            }else if ("*".equals(token)){
//                tmp1 = s.pop(); tmp2 = s.pop();
//                s.push(tmp1*tmp2);
//            }else if ("/".equals(token)){
//                tmp2 = s.pop(); tmp1 = s.pop();
//                s.push(tmp1/tmp2);
//            }else{
//                s.push(Integer.parseInt(token));
//            }
//        }
//        return s.peek();
//    }

    // leetcode 多数元素
//    public int majorityElement(int[] nums) {
//        int len = nums.length;
//        int res = nums[0];
//        HashMap<Integer, Integer> map = new HashMap<>();
//        for (Integer num : nums) {
//            if (!map.containsKey(num)) {
//                map.put(num, 1);
//                continue;
//            }
//            map.replace(num, map.get(num) + 1);
//            if (map.get(num) > len / 2) {
//                res = num;
//                break;
//            }
//        }
//        return res;
//    }

    // leetcode 621 任务调度器
    // 首先统计每种任务的数量，然后统计最多任务的种类（同时有几个任务数量都是最多)
    // 最后比较计算出来的最短执行时间和所有任务的总数量
    public int leastInterval(char[] tasks, int n) {
        //统计每种任务的数量
        int[] cnt = new int[26];
        // 统计任务数最多的任务的种类
        int maxcount = 0;
        // 任务最短执行时间
        int res;
        for (char task : tasks) {
            cnt[task - 'A']++;
        }
        sort(cnt);
        for (int i = 25; i >= 0; i--) {
            if (cnt[i] == cnt[25]) maxcount++;
            else break;
        }
        res = Math.max((cnt[25] - 1) * (n + 1) + maxcount, tasks.length);
        return res;
    }

    // leetcode 567 字符串的排列

    public boolean match(int[] cnt1, int[] cnt2) {
        for (int i = 0; i < cnt2.length; i++) {
            if (cnt1[i] != cnt2[i]) return false;
        }
        return true;
    }

    // 统计s2中同s1等长的滑动窗口，每次更小下一个字符，和过期的第一个字符的数量
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;
        int n1 = s1.length(), n2 = s2.length();
        int[] cnt1 = new int[26];
        int[] cnt2 = new int[26];
        // 对s2，只创建同s1等长的hash索引
        for (int i = 0; i < s1.length(); i++) {
            cnt1[s1.charAt(i) - 'a']++;
            cnt2[s2.charAt(i) - 'a']++;
        }

        for (int i = 0; i + n1 - 1 < n2; i++) {
            if (match(cnt1, cnt2)) return true;
            if (i + n1 >= n2) return false;  // 注意这里如果最后一个窗口不匹配，再往后加会导致越界，所以手动检测一下是否越界
            // 更新滑动窗口hash索引
            cnt2[s2.charAt(i) - 'a']--;
            cnt2[s2.charAt(i + n1) - 'a']++;
        }
        return false;
    }

    // leetcode 岛屿的最大面积
//    ！！！哥，是岛屿的最大面积，不是岛屿的数量，呜呜呜！！！
    public int checkIsland(int[][] grid, int i, int j) {
        if (i == -1 || i == grid.length || j == -1 || j == grid[0].length || grid[i][j] == 0) return 0;
        grid[i][j] = 0;
        int left = checkIsland(grid, i, j - 1);
        int right = checkIsland(grid, i, j + 1);
        int up = checkIsland(grid, i - 1, j);
        int down = checkIsland(grid, i + 1, j);
        return left + right + up + down + 1;
    }

    public int maxAreaOfIsland(int[][] grid) {
        int n = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 0)  //遇到海水，或者走过的土地(走过的土地已经反置为海水)，跳过检测
                    continue;
                n = Math.max(n, checkIsland(grid, i, j));
            }
        }
        return n;
    }

    // leetcode 最长连续递增序列，动态规划求解
    public int findLengthOfLCIS(int[] nums) {
        if (nums.length < 2) return nums.length;
        int[] len = new int[nums.length];
        len[0] = 1;
        int maxlen = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1])
                len[i] = len[i - 1] + 1;
            else len[i] = 1;
            maxlen = Math.max(len[i], maxlen);
        }
        return maxlen;
    }

    // x的root为f，查找并合并1的最终root
    // 比如 3的root为1，1的root为2,2的root为5,5的root为5,合并为3,1,2,5的root都是5
    public int findUnionSet(int[] s, int root) {
        int son, tmp;
        son = root;  // 保留末端节点
        // 查找最终root
        while (root != s[root]) {
            root = s[root];
        }
        while (son != root) {
            tmp = s[son];
            s[son] = root;
            son = tmp;
        }
        return root;  // 返回最终root
    }

    // leetcode 朋友圈
    public int findCircleNum(int[][] M) {
        int n = M.length;
        int count = n;
        int[] s = new int[n];
        for (int i = 0; i < n; i++) s[i] = i;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && M[i][j] > 0) {
                    // 合并两个root
                    if (s[i] != s[j]) {
                        int x = findUnionSet(s, i);
                        int y = findUnionSet(s, j);
                        if (x != y) { //root不同，总门派减一
                            s[x] = y;
                            count--;
                        }
                    }
                }
            }
        }
        return count;
    }

    // leetcode 最大正方形，动态规划求解
    public int maximalSquare(char[][] matrix) {
        if (matrix.length == 0) return 0;
        int w = matrix.length;
        int h = matrix[0].length;
        // 记录以当前元素为正方形右下角的最大正方形的边长
        int[][] areas = new int[w][h];
        // 最大正方形的边长
        int area = 0;

        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                if (i > 0 && j > 0 && matrix[i][j] == '1')
                    // 动态规划，如果当前元素为右下角能组成正方形
                    // 那么areas[i-1][j-1]，还有areas[i-1][j]和areas[i][j-1]，都是长度减1的小正方形
                    areas[i][j] = Math.min(Math.min(areas[i][j - 1], areas[i - 1][j]),
                            areas[i - 1][j - 1]) + 1;
                else areas[i][j] = matrix[i][j] - '0';
                if (areas[i][j] > area) area = areas[i][j];
                // System.out.println("areas["+i+","+j+"]="+areas[i][j]);
            }
        }
        return area * area;
    }

    // LeetCode 二进制求和
    public String addBinary(String a, String b) {
        // 把a当做大数，b当做小数
        if (a.length() < b.length()) {
            String tmp = a;
            a = b;
            b = tmp;
        }
        int c = 0;
        String ans = "";
        int i = a.length() - 1;
        for (int j = b.length() - 1; j >= 0; i--, j--) {
            int m = a.charAt(i) - '0';
            int n = b.charAt(j) - '0';
            int v = (m + n + c) % 2;
            if (m + n + c > 1) c = 1;
            else c = 0;
            ans = String.valueOf(v) + ans;
        }
        for (; i >= 0; i--) {
            int m = a.charAt(i) - '0';
            int v = (m + c) % 2;
            if (m + c > 1) c = 1;
            else c = 0;
            ans = String.valueOf(v) + ans;
        }
        if (c == 1) ans = "1" + ans;
        return ans;
    }

    public String convert(String s, int numRows) {

        int numCols = (numRows - 1) * ((int) (s.length() / (numRows + numRows - 2)) + 1);
        char[][] str = new char[numRows][numCols];
        // System.out.println(numCols);
        int k = 0, i = 0, j = 0;
        boolean f = true;  // 向下走的标记
        while (k < s.length()) {
            str[i][j] = s.charAt(k);
            if (f) {
                // 往下走，行加一，列不变
                i++;
                if (i == numRows - 1) f = false;
            } else {
                // 否则行减1，列加一
                i = (i - 1 + numRows) % numRows;
                j++;
                if (i == 0) f = true;
            }
            k++;
        }
        String res = "";
        for (i = 0; i < numRows; i++) {
            for (j = 0; j < numCols; j++) {
                if (str[i][j] != 0) res += str[i][j];
            }
        }
        return res;
    }

    public int fact(int n) {
        if (n <= 1)
            return 1;
        return n * fact(n - 1);
    }

    // leetcode 第k个排列
    public String getPermutation(int n, int k) {
        int fact_n = fact(n);
        // n! 倒序
        if (k == fact_n) {
            StringBuffer res = new StringBuffer();
            while (n > 0) res.append(n--);
            return res.toString();
        }
        fact_n = fact_n / n;  // (n-1)!
        // 用到的数字
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) nums.add(i);
        StringBuffer res = new StringBuffer();
        while (k >= 0) {
            // 这一位数字发生了排列
            // 基础知识注意啊 ！！！整型相除还是还是整型，要变成double
            if (k > fact_n && k % fact_n > 0) {
                int c = (int) Math.ceil((double) k / fact_n);
                res.append(nums.get(c - 1));
                nums.remove(c - 1);
                k = k % fact_n;
            } else if (k % fact_n == 0) {  // 后面是全排列
                int c = k / fact_n - 1;
                res.append(nums.get(c));
                nums.remove(c);
                for (int i = nums.size() - 1; i >= 0; i--)
                    res.append(nums.get(i));
                break;
            } else if (k == 1) {// 顺次取第一个数
                for (Integer num : nums)
                    res.append(num);
                break;
            } else if (k == 0) { // 倒序
                for (int i = nums.size() - 1; i >= 0; i--)
                    res.append(nums.get(i));
                break;
            } else {
                res.append(nums.get(0));
                nums.remove(0);
            }
            n--;
            fact_n /= n;
        }
        return res.toString();
    }

    // leetcode 杨辉三角II
    public List<Integer> getRow(int rowIndex) {
        List<Integer> row = new ArrayList<>();
        row.add(1);
        if (rowIndex == 0)
            return row;
        row.add(1);
        if (rowIndex == 1)
            return row;

        int n = 1;
        while (n < rowIndex) {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int i = 0; i < row.size() - 1; i++) {
                tmp.add(row.get(i) + row.get(i + 1));
            }
            tmp.add(1);
            row = tmp;
            n++;
        }
        return row;
    }

    // 两数之和 II - 输入有序数组：二分查找
//    public int[] twoSum(int[] numbers, int target) {
//        int[] res = new int[2];
//        int low = 0;
//        int high = numbers.length-1;
//        while(low < high){
//            if (numbers[low] + numbers[high] == target){
//                res[0] = low + 1;
//                res[1] = high + 1;
//                break;
//            }else if (numbers[low] + numbers[high] > target)
//                high--;
//            else low++;
//        }
//        return res;
//    }

    // LeetCode Excel表列数目：168 进制转换，注意从1开始，所以取余要减去1,整除也要减去1
    // 这样26整除，就是25/26=0，便会结束循环
    public String convertToTitle(int n) {
        StringBuffer res = new StringBuffer();
        while (n > 0) {
            n--;  // 从1开始，所以减去1
            res.append((char) (n % 26 + 'A'));
            n /= 26;
        }
        return res.reverse().toString();
    }

    // leetcode 同构字符串
    public boolean isIsomorphic(String s, String t) {
        HashMap<Character, Character> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (!map.containsKey(s.charAt(i))) {
                // 不包含key，包含value，多个字符对应同一字符
                if (map.containsValue(t.charAt(i)))
                    return false;
                map.put(s.charAt(i), t.charAt(i));
            } // 一个字符对应不同字符
            else if (map.get(s.charAt(i)) != t.charAt(i))
                return false;
        }
        return true;
    }

    // LeetCode easy 统计有序矩阵中的负数：利用本身有序性，从右上角开始遍历
    public int countNegatives(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int num = 0, k = 0;
        for (int j = n - 1; j >= 0; j--) {
            for (int i = k; i < m; i++) {
                if (grid[i][j] < 0) {
                    num += m - i;
                    i = m;
                } else {
                    k++;
                    if (k >= m) break;
                }
            }
        }
        return num;
    }

    // leetcode medium 最多可以参加的会议数目 贪心+排序
    // 将会议按照先结束排序，重载Arrays.sort函数
    public int maxEvents(int[][] events) {
        sort(events, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] != o2[1]) return o1[1] - o2[1];
                else return o1[0] - o2[0];
            }
        });
        boolean[] flag = new boolean[100005];
        int res = 0;
        // 对排序后的会议进行遍历
        for (int[] event : events) {
            // 先安排先结束的会议，从会议开始时间往后找，找到可用的一天为止
            for (int s = event[0]; s <= event[1]; s++) {
                if (!flag[s]) {
                    flag[s] = true;
                    res++;
                    break;
                }
            }
        }
        return res;
    }

    // leetcode hard 多次求和构造目标数组,采用优先级队列自动排序（注意是降序），寻找规律，倒推
    // 数组中总有一个元素为上一个更新所有元素的和，将其减去其他元素之和即上一次计算。
    public boolean isPossible(int[] target) {
        PriorityQueue<Long> pq = new PriorityQueue<>(new Comparator<Long>() {
            @Override
            public int compare(Long o1, Long o2) {
                return (int) (o2 - o1);
            }
        });
        long sum = 0;
        // 元素入队
        for (long t : target) {
            sum += t;
            pq.add(t);
        }
        while (true) {
            long maxv = pq.poll();
            if (maxv == 1)
                return true;
            // 不满足要求，和比其余元素小，或者出现负数
            if (maxv < 1 || maxv <= sum - maxv)
                return false;
            // 将最大值减去其余元素之和入队，更新sum
            pq.add(maxv - (sum - maxv));
            sum = maxv;
        }
    }

    // leetcode 供暖器
    public int findRadius(int[] houses, int[] heaters) {
        // 最小加热半径
        int r = 0;

        // 对热水器从小到大排序
        sort(heaters);
        // 当前房屋在当前记录的前后热水器之间
        for (int house : houses) {
            int left = 0, right = heaters.length - 1;
            int mid = 0;
            while (left < right) {
                mid = left + (right - left) / 2;
                if (heaters[mid] == house)
                    break;
                if (heaters[mid] > house) right = mid;
                else left = mid + 1;
            }
            if (heaters[mid] == house || heaters[left] == house) continue;
            if (heaters[left] < house || left == 0)
                r = Math.max(r, Math.abs(house - heaters[left]));
            else r = Math.max(r, Math.min(heaters[left] - house, house - heaters[left - 1]));
        }
        return r;
    }

    // LeetCode 俄罗斯信封套娃问题
    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes.length == 0) return 0;
        // 先按照w排序，w相同按照h降序排序
        // 避免w相同的信封使用多次
        sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]) return o1[0] - o2[0];
                return o2[1] - o1[1];
            }
        });

        // 保存一个最长上升子序列
        List<Integer> lis = new ArrayList<>();
        lis.add(envelopes[0][1]);

        // w排序后，取出h求最长上升子序列
        for (int i = 1; i < envelopes.length; i++) {
            int c = envelopes[i][1];
            // 相同则跳过
            if (c == envelopes[i - 1][1]) continue;
            // 大于最后一个元素，直接添加
            if (c > lis.get(lis.size() - 1))
                lis.add(c);
            else { // 二分查找比当前h大的最小元素，并替换
                int left = 0, right = lis.size() - 1;
                while (left < right) {
                    int mid = left + (right - left) / 2;
                    if (lis.get(mid) >= c) right = mid;
                    else left = mid + 1;
                }
                if (lis.get(left) != c) lis.set(left, c);
            }
        }
        return lis.size();
    }

    // leetcode 130 被围绕的区域
    public void dfs(char[][] board, char oldc, char newc, int r, int c) {
        // 越界或者走过当前点，直接返回
        if (r < 0 || r >= board.length || c < 0 || c >= board[0].length)
            return;
        if (board[r][c] != oldc) return;
        board[r][c] = newc;
        // 向四方寻找0如果到达边界的上一个，不在递归寻找
        dfs(board, oldc, newc, r, c - 1);
        dfs(board, oldc, newc, r, c + 1);
        dfs(board, oldc, newc, r - 1, c);
        dfs(board, oldc, newc, r + 1, c);
    }

    public void solve(char[][] board) {
        // 处理边界，从边界开始DFs，将和边界'O'相连的点标记为'F'
        // 处理上下两行
        for (int c = 0; c < board[0].length; c++) {
            dfs(board, 'O', 'F', 0, c);
            dfs(board, 'O', 'F', board.length - 1, c);
        }
        // 处理左右两列
        for (int r = 1; r < board.length - 1; r++) {
            dfs(board, 'O', 'F', r, 0);
            dfs(board, 'O', 'F', r, board[0].length - 1);
        }
        // 处理内部点，将'O'替换成'X'
        for (int r = 1; r < board.length - 1; r++)
            for (int c = 1; c < board[0].length - 1; c++)
                dfs(board, 'O', 'X', r, c);

        //再次处理边界，将'F'替换成'O'
        // 处理上下两行
        for (int c = 0; c < board[0].length; c++) {
            dfs(board, 'F', 'O', 0, c);
            dfs(board, 'F', 'O', board.length - 1, c);
        }
        // 处理左右两列
        for (int r = 1; r < board.length - 1; r++) {
            dfs(board, 'F', 'O', r, 0);
            dfs(board, 'F', 'O', r, board[0].length - 1);
        }
    }

    // leetcode 键盘行
    public String[] findWords(String[] words) {
        Map<Character, Integer> map = new HashMap<>();
        char[][] r1 = {
                {'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'},
                {'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'},
                {'Z', 'X', 'C', 'V', 'B', 'N', 'M'}
        };
        for (int i = 0; i < r1.length; i++)
            for (char c : r1[i]) {
                map.put(c, i);
                // 大写+32即为小写
                map.put((char) (c + 32), i);
            }

        List<String> res = new ArrayList<>();
        for (String word : words) {
            boolean flag = true;
            for (int i = 1; i < word.length(); i++) {
                if (map.get(word.charAt(i)) != map.get(word.charAt(i - 1))) {
                    flag = false;
                    break;
                }
            }
            if (flag) res.add(word);
        }
        String[] ans = new String[res.size()];
        for (int i = 0; i < res.size(); i++)
            ans[i] = res.get(i);
        return ans;
    }

    // leetcode 数字的补数
    // 5 0101 1111 - 0101 = 1010
    // 所以拿到1111 然后和5异或
    public int findComplement(int num) {
        int sum = 1;
        int tmp = num;
        while (tmp > 0) {
            sum <<= 1; // sum左移乘以2
            tmp >>= 1; // tmp右移
        }
        sum -= 1;
        return sum ^ num;
    }

    // leetcode 单词接龙 双端BFS 广度优先搜索
    // 比较两个单词是否只有一个字母不同
    public boolean canConvert(String s, String t) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != t.charAt(i))
                count++;
            if (count > 1) return false;
        }
        return true;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        // 保存单词访问记录
        boolean[] visited1 = new boolean[wordList.size()];
        boolean[] visited2 = new boolean[wordList.size()];
        // 如果存在，则将beginWord endWord标为true
        int idx = wordList.indexOf(beginWord);
        if (idx != -1) visited1[idx] = true;
        idx = wordList.indexOf(endWord);
        visited2[idx] = true;
        // 分别保存begin,start走过的单词
        Queue<String> queue1 = new LinkedList<>();
        queue1.offer(beginWord);
        Queue<String> queue2 = new LinkedList<>();
        queue2.offer(endWord);
        int count = 0;
        // 一直找直到队列为空
        while (!queue1.isEmpty() && !queue2.isEmpty()) {
            count++;
            // 每一次从长度短的队列遍历
            int size1 = queue1.size();
            int size2 = queue2.size();
            if (size1 > size2) {
                // 交换两个队列和访问数组
                Queue<String> tmp = queue1;
                queue1 = queue2;
                queue2 = tmp;
                boolean[] t = copyOf(visited1, visited1.length);
                visited1 = visited2;
                visited2 = t;
            }
            size1 = queue1.size();
            // 每一次遍历队列，找到每个单词的next 都加到队列中
            while (size1-- > 0) {
                String s = queue1.poll();
                for (int i = 0; i < wordList.size(); i++) {
                    String word = wordList.get(i);
                    // 不能转换 或者访问过
                    if (!canConvert(s, word) || visited1[i]) continue;
                    // 两个队列碰头，则返回
                    if (visited2[i]) return count + 1;
                    // System.out.println("---" + s + "---" + word);
                    queue1.offer(word);
                    visited1[i] = true;
                }
            }
        }
        return 0;
    }

    // leetcode 加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int gases = 0;
        int costs = 0;
        int len = gas.length;
        // 保存可以开始的下标
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < len; i++) {
            gases += gas[i];
            costs += cost[i];
            if (gas[i] >= cost[i])
                set.add(i);
        }
        if (gases < costs)  // 总消耗大于汽油量
            return -1;
        Iterator<Integer> itr = set.iterator();
        while (itr.hasNext()) {
            int start = (int) itr.next();
            int last = gas[start];
            // 从start出发，往后走,走len-1站
            for (int i = 1; i < len; i++) {
                // 到达某站后的剩余汽油量
                last = last + gas[(i + start) % len] - cost[(i + start - 1) % len];
                // 如果剩余量小于cost，表示当前路线不可达，退出循环
                if (last < cost[(i + start) % len]) {
//                    itr.remove();
                    break;
                }
            }
            // 最后回到start，如果剩余量足够则返回start
            if (last >= cost[start]) {
                return start;
            }
//            else itr.remove();
        }
        return -1;
    }

    // leetcode 课程表，拓扑排序
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Queue<Integer> queue = new LinkedList<>();
        // 记录每个节点的入度
        int[] inDegree = new int[numCourses];
        // 邻接链表
        int[][] w = new int[numCourses][numCourses];
        // 初始化入度列表和邻接链表
        for (int[] pre : prerequisites) {
            inDegree[pre[0]]++;
            w[pre[1]][pre[0]] = 1;
        }
        // 记录入队次数
        int num = 0;
        // 将入度为0的节点入队
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
                // System.out.println("initial---" + i + "---");
            }
        }
        while (!queue.isEmpty()) {
            // 队首元素出队
            int q = queue.poll();
            // System.out.println("poll---" + q + "---");
            num++;
            // 对该元素发出边的所有节点，入度减1，如果减完之后入度为0，入队
            for (int i = 0; i < numCourses; i++) {
                if (w[q][i] == 1) {
                    inDegree[i]--;
                    if (inDegree[i] == 0)
                        queue.offer(i);
                }
            }
        }
        // 如果入队次数为numCourses，返回true
        return num == numCourses;
    }

    // leetcode 完全平方数
    // 动态规划
    public int numSquares(int n) {
        if (n < 4) return n;
        int m = (int) Math.sqrt((double) n);
        // 能够完全开平方
        if (m * m == n) return 1;
        // dp[i] 表示组成i的完全平方数的个数
        int[] dp = new int[n + 1];
        // dp[i]最坏是i=i*1，出现的最大的完全平方数为i开方向下取整
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            // 遍历组成i的完全平方数 从1到最大的j的平方,取最小值
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
            // System.out.println("dp[" + i+ "]=" + dp[i]);
        }
        return dp[n];
    }

    // leetcode 动态规划 + 递归 + 栈
    // 通过递归记录生成分割方案的过程，用栈保存分割方案
    // 用递归来模拟生成分割方案的过程
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        if (s.length() == 0) return res;
        int len = s.length();
        // 记录flag[i][j]是否为回文串
        boolean[][] flag = new boolean[len][len];
        // 动态规划生成flag，l为长度
        for (int l = 1; l <= len; l++)
            for (int i = 0; i + l - 1 < len; i++) {
                if (l == 1) {
                    flag[i][i] = true;
                    continue;
                }
                // 两端相同
                if (s.charAt(i) == s.charAt(i + l - 1) && (l == 2 || flag[i + 1][i + l - 2]))
                    flag[i][i + l - 1] = true;
                // System.out.println("flag[" + i + "]" + "[" + (i+l-1) + "]=" + flag[i][i+l-1]);
            }
        // 保存生成记录
        Deque<String> stack = new ArrayDeque<>();
        backtracking(s, 0, len, res, stack, flag);
        return res;
    }

    public void backtracking(String s, int start, int end, List<List<String>> res, Deque<String> stack, boolean[][] flag) {
        if (start == end) {
            // 遍历结束，将栈中内容加到res中
            res.add(new ArrayList<>(stack));
            return;
        }
        // 取start到i为前缀
        for (int i = start; i < end; i++) {
            // 截取前缀不是回文串，剪枝
            if (!flag[start][i])
                continue;
            stack.addLast(s.substring(start, i + 1));
            backtracking(s, i + 1, end, res, stack, flag);
            stack.removeLast();
        }
    }

    // leetcode 54 螺旋矩阵
    // 每一圈都是顺时针遍历，注意边界条件！
    // 上边界走完，走右边界时候，起始行索引是up+1，同样走下边界时候，起始列索引是right-1
    // 最后左边界，是down-1！！！你个大傻子！！！
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix.length == 0) return new ArrayList<Integer>();
        // 记录四个边界
        int left = 0, right = matrix[0].length - 1, up = 0, down = matrix.length - 1;
        // 下标索引
        int i = 0, j = 0;
        List<Integer> list = new ArrayList<>();
        while (left <= right && up <= down) {
            // 在上边界上，往右走
            for (j = left; j <= right; j++) {
                list.add(matrix[up][j]);
            }
            // 在右边界上，往下走
            for (i = up + 1; i <= down; i++) {
                list.add(matrix[i][right]);
            }
            // 防止只有一行或一列的情况
            if (left < right && up < down) {
                // 在下边界上，往左走
                for (j = right - 1; j >= left; j--) {
                    list.add(matrix[down][j]);
                }
                // 在左边界上，往上走
                for (i = down - 1; i > up; i--) {
                    list.add(matrix[i][left]);
                }
                // System.out.println("i="+i+",j="+j);
            }
            left++;
            right--;
            up++;
            down--;
        }
        return list;
    }

    // leetcode 91 解码方法，动态规划
    public int numDecodings(String s) {
        // 长度为0或起始字符为0，直接返回0
        if (s.length() == 0 || s.charAt(0) == '0')
            return 0;
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        dp[1] = 1;
        // 转移状态方程 dp[i] = dp[i-1] + dp[i-2]
        // 对于前i个字符，第i个字符肯定能编码（不为0的话），后面两个字符可能能够编码（小于26）
        // 不存在三个字符能编码的情况，所以考虑最后两个字符足矣
        for (int i = 2; i <= len; i++) {
            // 字符为0，前面只能是1或2
            if (s.charAt(i - 1) == '0') {
                if (s.charAt(i - 2) == '1' || s.charAt(i - 2) == '2')
                    dp[i] = dp[i - 2];
                else return 0;
            } // 前一位不为0，且最后两个字符对应数字不大于26，后两位字符能编码
            else if (s.charAt(i - 2) != '0' && (s.charAt(i - 2) - '0') * 10 + s.charAt(i - 1) - '0' <= 26)
                dp[i] = dp[i - 1] + dp[i - 2];
            else dp[i] = dp[i - 1];
        }
        return dp[len];
    }

    // leetcode 179 最大数
    public String largestNumber(int[] nums) {
        if (nums.length == 0)
            return "";
        if (nums.length == 1)
            return String.valueOf(nums[0]);
        List<String> strs = new ArrayList<>();
        for (int num : nums)
            strs.add(String.valueOf(num));
        Collections.sort(strs, new Comparator<String>() {
            // 比较两个字符串如何拼接，字典序更大，此为本题之精妙之处。
            @Override
            public int compare(String o1, String o2) {
                String a = o1 + o2;
                String b = o2 + o1;
                // 降序排列
                return b.compareTo(a);
            }
        });
        StringBuffer res = new StringBuffer();
        for (String str : strs)
            res.append(str);
        String s = res.toString();
        // 以0开头，只保留一个0
        if (s.startsWith("0")) {
            int i = 0;
            while (i < s.length() && s.charAt(i) == '0') i++;
            s = s.substring(i - 1);
        }
        return s;
    }

    // leetcode 152. 乘积最大子序列，动态规划
    // 动态规划，记录以当前下标i结尾的子数组的最大乘积，在遍历下一个元素的时候，可能遇到负数，这样之前最大的就变成了最小的，最小的就变成了最大的，所以还需要维护以当前下标i结尾的子数组的最小乘积
    // 如果遇到负数，则交换最大与最小，并且在过程中不断更新最大值
    public int maxProduct(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        int len = nums.length;
        // 维护一个最大乘积数组和一个最小乘积数组，分别存取以i下标元素结尾的子序列的最大乘积
        // 因为可能有负数，所以上一个mindp乘以当前的负数可能是最大乘积，所以比较三个数即可。
        // 优化之后维护三个数即可
        int mindp, maxdp, res;
        mindp = maxdp = res = nums[0];
        for (int i = 1; i < len; i++) {
            if (nums[i] == 0) {
                mindp = maxdp = 0;
                continue;
            }
            // 负数交换最大最小值
            if (nums[i] < 0) {
                int tmp = mindp;
                mindp = maxdp;
                maxdp = tmp;
            }
            mindp = Math.min(mindp * nums[i], nums[i]);
            maxdp = Math.max(maxdp * nums[i], nums[i]);
            res = Math.max(maxdp, res);
        }
        return res;
    }


//    public int maxProduct(int[] nums) {
//        if (nums.length == 1)
//            return nums[0];
//        int len = nums.length;
//        // 维护一个最大乘积数组和一个最小乘积数组，分别存取以i下标元素结尾的子序列的最大乘积
//        // 因为可能有负数，所以上一个mindp乘以当前的负数可能是最大乘积，所以比较三个数即可。
//        int[] mindp = new int[len];
//        int[] maxdp = new int[len];
//        mindp[0] = maxdp[0] = nums[0];
//        int res = nums[0];
//        for (int i = 1; i < len; i++){
//            mindp[i] = Math.min(mindp[i-1]*nums[i], Math.min(nums[i], maxdp[i-1]*nums[i]));
//            maxdp[i] = Math.max(mindp[i-1]*nums[i], Math.max(nums[i], maxdp[i-1]*nums[i]));
//            res = Math.max(maxdp[i], res);
//        }
//        return res;
//    }

    // leetcode 基本计算器II
    public int compute2Num(int n1, int n2, char op) {
        if (op == '+')
            return n1 + n2;
        if (op == '-')
            return n1 - n2;
        if (op == '*')
            return n1 * n2;
        return n1 / n2;
    }

    public int calculate(String s) {
        Stack<Integer> nums = new Stack<>();
        Stack<Character> ops = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // 遇到数字，往后找
            if (c >= '0' && c <= '9') {
                int num = c - '0';
                i++;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    i++;
                }
                nums.push(num);
                i--;
            } else if (c == '*' || c == '/') {
                if (!ops.isEmpty()) {
                    char op = ops.peek();
                    if (op == '*' || op == '/') {
                        op = ops.pop();
                        int n2 = nums.pop();
                        int n1 = nums.pop();
                        nums.push(compute2Num(n1, n2, op));
                    }

                }
                ops.push(c);  // 乘除号入栈
            } else if (c == '+' || c == '-') {
                // 遇到加减号，如果符号栈中有符号，计算之后结果入栈，op入栈，无符号则直接入栈
                while (!ops.isEmpty()) {
                    char op = ops.pop();
                    int n2 = nums.pop();
                    int n1 = nums.pop();
                    nums.push(compute2Num(n1, n2, op));
                }
                ops.push(c);
            }
        }
        // 处理剩余运算
        while (!ops.isEmpty()) {
            char op = ops.pop();
            int n2 = nums.pop();
            int n1 = nums.pop();
            nums.push(compute2Num(n1, n2, op));
        }
        return nums.pop();
    }

    // leetcode 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        res[0] = 1;
        int k = 1;
        // 除nums[i]之外其余个元素的乘积 = 当前元素左边元素的乘积 * 当前元素右边元素的乘积
        // 求左边元素乘积
        for (int i = 1; i < nums.length; i++)
            res[i] = res[i - 1] * nums[i - 1];
        // 求右边元素乘积
        for (int i = nums.length - 2; i >= 0; i--) {
            k = k * nums[i + 1];
            res[i] = res[i] * k;
        }
        return res;
    }

    // 454. 四数相加 II 查找表
    // O(n2) 遍历A和B任意元素之和存入查找表，然后遍历C和D，如果查找表中存在-(C[k]+D[l])，加到结果中
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int N = A.length;
        Map<Integer, Integer> abmap = new HashMap<>();

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                int num = abmap.getOrDefault(A[i] + B[j], 0);
                abmap.put(A[i] + B[j], num + 1);
            }

        int res = 0;
        for (int k = 0; k < N; k++)
            for (int l = 0; l < N; l++) {
                int sum = -(C[k] + D[l]);
                if (abmap.containsKey(sum)) {
                    res += abmap.get(sum);
                }
            }
        return res;
    }

    // 395. 至少有K个重复字符的最长子串
    // 递归+分治法
    public int longestSubstr(String s, int k, int left, int right) {
        if (right - left + 1 < k)
            return 0;
        if (s.length() < k)
            return 0;
        // 保存z子串内每个字符出现次数
        Map<Character, Integer> map = new HashMap<>();
        for (int i = left; i <= right; i++) {
            int num = map.getOrDefault(s.charAt(i), 0);
            map.put(s.charAt(i), num + 1);
        }
        // 处理左右边界，如果出现次数小于k，以此为分隔符，递归左右两个子串的最长子串,返回最大值
        while (right - left + 1 >= k && map.get(s.charAt(left)) < k) left++;
        while (right - left + 1 >= k && map.get(s.charAt(right)) < k) right--;
        for (int i = left; i <= right; i++) {
            if (map.get(s.charAt(i)) < k) {
                return Math.max(longestSubstr(s, k, left, i - 1),
                        longestSubstr(s, k, i + 1, right));
            }
        }
        return right - left + 1;
    }

    public int longestSubstring(String s, int k) {
        if (s.length() < k)
            return 0;
        return longestSubstr(s, k, 0, s.length() - 1);
    }

    // 1013. 将数组分成和相等的三个部分
    // 首先计算数组 A中所有数字总和 sum
    //遍历数组 A 查找和为 sum / 3 的子数组个数
    //如果找到了三个这样的子数组则返回 true， 找不到三个就返回 falsefalse
    public boolean canThreePartsEqualSum(int[] A) {
        int[] sum = new int[A.length];
        sum[0] = A[0];
        for (int i = 1; i < A.length; i++) {
            sum[i] = sum[i - 1] + A[i];
        }
        if (sum[A.length - 1] % 3 != 0)
            return false;
        int n = sum[A.length - 1] / 3;
        for (int i = 0; i < A.length; i++) {
            if (sum[i] == n) {
                int k = i + 1;
                // 必须保证能分成三个区间
                while (k < A.length - 1) {
                    if (sum[k] - sum[i] == n)
                        return true;
                    k++;
                }
                return false;
            }
        }
        return false;
    }

    // 1071. 字符串的最大公因子
//    public String gcdOfStrings(String str1, String str2) {
//        String res = "";
//        String both = "", last = "";
//        if (str1.length() >= str2.length()){
//            int idx = str1.indexOf(str2);
//            // 如果str2不是长的str1的子串，或者不是从0开始相等，返回空串
//            if (idx == -1 || idx != 0)
//                return "";
//            // 长度相等
//            if (str1.length() == str2.length() )
//                return str1;
//            both = str2;
//            last = str1.substring(str2.length());
//        }else{
//            int idx = str2.indexOf(str1);
//            if (idx == -1 || idx != 0)
//                return "";
//            both = str1;
//            last = str2.substring(str1.length());
//        }
//        return gcdOfStrings(both, last);
//    }
    public String gcdOfStrings(String str1, String str2) {
        if (!(str1 + str2).equals(str2 + str1))
            return "";
        return str1.substring(0, gcd(str1.length(), str2.length()));
    }

    public int gcd(int a, int b) {
        if (b == 0)
            return a;
        return gcd(b, a % b);
    }

    // 面试题 17.10. 主要元素
    // 摩尔投票法 + 验证
    public int majorityElement(int[] nums) {
        int count = 0;
        int tmp = nums[0];
        // 摩尔投票法，求出现次数超过半数的元素，前提是一定有超半数元素在。
        // 如果当前元素等于标记元素，计数+1，否则计数-1，
        // 如果计数为0，说明之前的标记元素出现次数和其它元素出现次数抵消了，标记当前元素为新的标记元素，重新开始计数。
        // 因为超过半数的元素可以把其它所有的数都抵消，最后剩下的一定是众数元素。
        for (int num : nums) {
            if (num == tmp)
                count++;
            else count--;
            if (count <= 0) {
                tmp = num;
                count = 1;
            }
        }
        int n = nums.length / 2;
        int t = 0;
        for (int num : nums) {
            if (num == tmp)
                t++;
            if (t > n)
                return tmp;
        }
        return -1;
    }

    // 239.滑动窗口最大值
    // 双端队列，注意队列中保存的是元素下标
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (k == 0 || nums.length == 0)
            return new int[0];
        if (k == 1)
            return nums;
        Deque<Integer> dq = new LinkedList<>();
        int[] res;
        // 窗口比总长度长时
        if (k >= nums.length) res = new int[1];
        else res = new int[nums.length - k + 1];
        int p = 0;
        for (int i = 0; i < nums.length; i++) {
            // 将新添加的元素和队尾元素比较，比当前值小的，全部出队
            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i])
                dq.pollLast();
            dq.addLast(i);
            // 判断队首元素是否在滑动窗口内
            if (dq.peekFirst() < i - k + 1)
                dq.pollFirst();
            // i达到第一个滑动窗口长度后，写入res
            if (i >= k - 1) res[p++] = nums[dq.peekFirst()];
        }
        return res;
    }

    // 面试题57 - II. 和为s的连续正数序列
    // 滑动窗口法
    public int[][] findContinuousSequence(int target) {
        List<int[]> res = new ArrayList<>();
        int l = 1, r = 1;
        int sum = 1;
        while (l <= target / 2) {
            if (sum == target) { //找到一个序列
                int[] seq = new int[r - l + 1];
                for (int i = l; i <= r; i++) seq[i - l] = i;
                res.add(seq);
                sum -= l;
                l++;
            } else if (sum < target) {
                r++;
                sum += r;
            } else {
                sum -= l;
                l++;
            }
            // System.out.println("l="+l+",r="+r+",sum="+sum);
        }
        return res.toArray(new int[res.size()][]);
    }

    // 竞赛题：矩阵中的幸运数
    public List<Integer> luckyNumbers(int[][] matrix) {
        List<Integer> luckynums = new ArrayList<>();
        int m = matrix.length, n = matrix[0].length;
        int[] minRow = new int[m];
        int[] maxCol = new int[n];
        // 找到每一列最大值
        for (int j = 0; j < n; j++) {
            maxCol[j] = 0;
            for (int i = 1; i < m; i++) {
                if (matrix[i][j] > matrix[maxCol[j]][j]) {
                    maxCol[j] = i;
                }
            }
        }
        // 找到每一行最小值
        for (int i = 0; i < m; i++) {
            // 同行最小的idx
            minRow[i] = 0;
            // 遍历一行找到最小值
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] < matrix[i][minRow[i]]) {
                    minRow[i] = j;
                }
            }
        }
        // 看是否同行最小的元素也是同列最大的元素
        for (int i = 0; i < m; i++) {
            int col = minRow[i];
            if (maxCol[col] == i)
                luckynums.add(matrix[i][col]);
        }
        return luckynums;
    }

    // 竞赛题：最大团队表现值，超时
    // 保存最大团队表现值
    int performance = 0;

    // dfs ， c 当前工程师人数，s 已有speed之和，e 已有最低效率
    public void dfs(int n, int[] speed, int[] efficiency, boolean[] used, int p, int s, int e, int k, int c) {
        if (c > k) return;
        if (p > performance) performance = p;
        for (int i = 0; i < n; i++) {
            if (used[i]) continue;
            used[i] = true;
            // 计算新的p
            p = (s + speed[i]) * Math.min(e, efficiency[i]);
            dfs(n, speed, efficiency, used, p, s + speed[i], Math.min(e, efficiency[i]), k, c + 1);
            used[i] = false;
        }
    }

    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        if (k == 1 && n == 1) return speed[0] * efficiency[0];
        boolean[] used = new boolean[n];
        dfs(n, speed, efficiency, used, 0, 0, Integer.MAX_VALUE, k, 0);
        return performance;
    }

    // 994.腐烂的橘子 BFS + 队列
    public int orangesRotting(int[][] grid) {
        int M = grid.length, N = grid[0].length;
        // 记录新鲜橘子数量
        int count = 0;
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                // 计数新鲜橘子，将腐烂橘子坐标加到队列中
                if (grid[i][j] == 1)
                    count++;
                if (grid[i][j] == 2) {
                    q.offer(new int[]{i, j});
                }
            }
        int bound = 0;  // 感染轮次
        // 有新鲜橘子，并且队列中有坏橘子
        while (count > 0 && !q.isEmpty()) {
            bound++;
            int n = q.size();  // 当前轮次坏橘子数量
            // 遍历当前轮次的坏橘子，感染新鲜橘子
            for (int i = 0; i < n; i++) {
                int[] loc = q.poll();
                int r = loc[0], c = loc[1];
                // 向四周感染橘子,发现了四周的新鲜橘子，就将计数-1，感染之，加入感染队列
                if (c > 0 && grid[r][c - 1] == 1) {
                    count--;
                    grid[r][c - 1] = 2;
                    q.offer(new int[]{r, c - 1});
                }
                if (c < N - 1 && grid[r][c + 1] == 1) {
                    count--;
                    grid[r][c + 1] = 2;
                    q.offer(new int[]{r, c + 1});
                }
                if (r > 0 && grid[r - 1][c] == 1) {
                    count--;
                    grid[r - 1][c] = 2;
                    q.offer(new int[]{r - 1, c});
                }
                if (r < M - 1 && grid[r + 1][c] == 1) {
                    count--;
                    grid[r + 1][c] = 2;
                    q.offer(new int[]{r + 1, c});
                }
            }
        }
        // 仍有新鲜橘子残留
        if (count > 0)
            return -1;
        return bound;
    }

    // 5178. 四因数
    public int sumFourDivisors(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            int res = 0;
            // 统计因子数目
            int cnt = 0;
            int p = (int) Math.sqrt((double) num);
            // 遍历能否被整除
            for (int j = 1; j <= p; j++) {
                if (cnt > 4)
                    break;
                // 找到因子
                if (num % j == 0) {
                    // 统计因子之和，注意，这里要区分因子和商是否相等
                    int other = num / j;
                    if (j != other) {
                        res += j + other;
                        cnt += 2;
                    } else {
                        res += j;
                        cnt += 1;
                    }
                }
            }
            // 完全平方数一定不是四个因数 例如16：1,2,4,8,16
            if (cnt == 4 && p * p != num)
                ans += res;
        }
        return ans;
    }

    // 1392. 最长快乐前缀
    // 字符串哈希
    public String longestPrefix(String s) {
        if (s.length() <= 1)
            return "";
        int len = s.length();
        final int p = 131;
        final long M = (long) (1e9 + 7);
        long[] hash = new long[len + 1];
        long[] power = new long[len + 1];
        char[] str = s.toCharArray();

        hash[0] = 0;
        power[0] = 1;
        for (int i = 1; i <= len; i++) {
            hash[i] = (hash[i - 1] * p + (str[i - 1] - 'a' + 1));
            power[i] = (power[i - 1] * p);
        }

        for (int i = len - 1; i >= 1; i--) {
            long prefix = hash[i];
            long postfix = hash[len] - hash[len - i] * power[i];
//            System.out.println(prefix +" " + postfix);
            if (prefix == postfix) {
                return s.substring(0, i);
            }
        }
        return "";
    }

    // 面试题40. 最小的k个数, 最大堆解法，时间复杂度 O(Nlogk),空间复杂度 O(k)
//    public int[] getLeastNumbers(int[] arr, int k) {
//        if (k == 0 || arr.length == 0)
//            return new int[0];
//        Queue<Integer> pq = new PriorityQueue<>(k, (v1, v2)->(v2-v1));
//        for (int a: arr){
//            if (pq.size() < k){
//                pq.offer(a);
//            }else if (a < pq.peek()){
//                pq.offer(a);
//                pq.poll();
//            }
//        }
//
//        int[] ans = new int[k];
//        int i = 0;
//        while(!pq.isEmpty()){
//            ans[i] = pq.poll();
//            i++;
//        }
//        return ans;
//    }


    public int[] topKSelect(int[] num, int k, int l, int r) {
        int pos = partition(num, l, r);
        // 前k个最小的数正好是pos左边的数组,或者加上当前pos位置的数
        if (pos == k || pos == k - 1)
            return copyOf(num, k);
            // 左边数组多余k个，递归快速选择左边数组
        else if (pos > k)
            return topKSelect(num, k, l, pos - 1);
            // 左边数组不足k个，继续递归选择右边数组
            // 注意虽然划分的是[pos+1, r]区间，但是返回的下标总是包含大于pos的，所以这里参数还是k
        else return topKSelect(num, k, pos + 1, r);
    }

    // 快速排序的划分函数，return j 使得
    // num[low] <= ... <= num[j-1] < num[j] < num[j+1] <= ... <= num[high]
    public int partition(int[] num, int low, int high) {
        int i = low, j = high;
        int v = num[i];
        while (true) {
            while (num[++i] < v && i <= high) ;
            while (num[--j] > v && j >= low) ;
            if (i >= j)
                break;
            // swap(num, i, j)
            int tmp = num[i];
            num[i] = num[j];
            num[j] = tmp;
        }
        num[low] = num[j];
        num[j] = v;
        return j;
    }

    // 面试题40. 最小的k个数, 最大堆解法，时间复杂度 O(Nlogk),空间复杂度 O(k)
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr.length == 0 || k == 0)
            return new int[0];
        return topKSelect(arr, k, 0, arr.length - 1);
    }

    public int max_gcd(int x, int y) {
        if (y == 0)
            return x;
        if (x == 1 || y == 1)
            return 1;
        return max_gcd(y, x % y);
    }

    // 365. 水壶问题,裴蜀定理，对于整数x,y 使得 ax + by = z成立的整数a，b存在的条件是z能除开x和y的最大公约数
    public boolean canMeasureWater(int x, int y, int z) {
        if (x + y < z)
            return false;
        if (x == 0 || y == 0)
            return z == 0 || x + y == z;
        return z % max_gcd(x, y) == 0;
    }

    // 914. 卡牌分组
    // 统计每个数字出现的次数，求最大公约数
    public boolean hasGroupsSizeX(int[] deck) {
        int[] count = new int[10001];
        Set<Integer> nums = new HashSet<>();
        for (int num : deck) {
            count[num]++;
            nums.add(num);
        }
        int gcd = 0;
        for (Integer num : nums) {
            if (gcd == 0) {
                gcd = count[num];
                continue;
            }
            int t = max_gcd(gcd, count[num]);
            if (t == 1)
                return false;
            gcd = t;
        }
        // gcd等于1，表明只有一个数字
        return gcd != 1;
    }

    // 945. 使数组唯一的最小增量
    // 方法1：排序 时间复杂度 O(nlogn)
    // 这里隐含了贪心思想，每个重复数字尽可能变成最近的更大的数字
//    public int minIncrementForUnique(int[] A) {
//        if (A.length < 2)
//            return 0;
//        // 按照升序排序
//        Arrays.sort(A);
//        int move = 0;
//        for (int i = 1; i < A.length; i++){
//            // 如果有重复的数，或者比前一个小（此时是前边数字+1导致的）
//            // 就把当前的val更新为上一个元素+1，move响应更新
//            if (A[i] <= A[i-1]){
//                move += A[i-1] - A[i] + 1;
//                A[i] = A[i-1] + 1;
//            }
//        }
//        return move;
//    }

    // 方法2：计数排序 时间复杂度 O(n+k)
//    public int minIncrementForUnique(int[] A) {
//        int[] count = new int[40001];
//        int upper = 0;
//        //统计每个数字出现的次数,记录出现的最大数字
//        for (int a: A){
//            count[a]++;
//            upper = Math.max(upper, a);
//        }
//
//        int move = 0;
//        // 遍历每个数字，如果出现次数大于1，就要变成下一个数字,下一个数字出现次数对应增加
//        for (int i = 1; i < upper; i++){
//            if (count[i] > 1){
//                count[i+1] += count[i] - 1;
//                move += count[i] - 1;
//            }
//        }
//
//        // 单独处理最大的， 因为如果最大的是39999，move之后有可能会变成超过40000的数字
//        if (count[upper] > 1){
//            int d = count[upper]-1;
//            // move之后，变成，upper+1, upper+2, ... , upper + d
//            // 用等差数列计算
//            move += d*(1+d)/2;
//        }
//        return move;
//    }

    // 方法3：线性探查+路径压缩，时间复杂度 O(n)
    public int minIncrementForUnique(int[] A) {
        if (A.length < 2)
            return 0;
        // 保存线性hash表，初始化为-1
        int[] pos = new int[80001];
        fill(pos, -1);
        int move = 0;
        for (int a : A) {
            // 如果a位置为空，之间放置a
            if (pos[a] == -1) {
                pos[a] = a;
            } else {
                // 否则查找能放置a的位置b，并更新move
                int b = findPos(pos, a);
                pos[a] = b;
                move += b - a;
            }
        }
        return move;
    }

    // hash线性探查
    public int findPos(int[] pos, int a) {
        if (pos[a] == -1) {
            pos[a] = a;
            return a;
        }
        // 如果a位置已经有数字放置，从上一次探查的最后位置+1开始探查
        int b = pos[a] + 1;
        int t = findPos(pos, b);
        pos[a] = t;  // 寻址后的新空位要重新赋值给pos[a]哦（路径压缩）
        return t;
    }

    // 820. 单词的压缩编码，比较后缀法
//    public int minimumLengthEncoding(String[] words) {
//        Set<String> codes = new HashSet<>();
//        int size = 0;
//        // 按照单词倒序的字典序排序
//        Arrays.sort(words, new Comparator<String>() {
//            @Override
//            public int compare(String o1, String o2) {
//                String o1r = new StringBuffer(o1).reverse().toString();
//                String o2r = new StringBuffer(o2).reverse().toString();
//                return o1r.compareTo(o2r);
//            }
//        });
//        for (int i = 0; i < words.length-1; i++){
//            // 如果下一个词以当前词结尾，删除当前词，添加下一个词
//            if (words[i+1].endsWith(words[i])){
//                if (codes.contains(words[i])) {
//                    codes.remove(words[i]);
//                    size -= words[i].length();
//                }
//                codes.add(words[i+1]);
//                size += words[i+1].length();
//            // 否则如果编码单词中不存在当前词，则添加当前词
//            }else if (!codes.contains((words[i]))){
//                codes.add(words[i]);
//                size += words[i].length();
//            }
//        }
//        // 单独判断最后一个词
//        if (!codes.contains(words[words.length-1])){
//            codes.add(words[words.length-1]);
//            size += words[words.length-1].length();
//        }
//        // 加上#号长度
//        return size+codes.size();
//    }

    // 820. 单词的压缩编码，字典树法
    // 字典树的节点类，包含一个children数组，长为26
    class TrieNode {
        TrieNode[] children;

        public TrieNode() {
            children = new TrieNode[26];
        }
    }

    // 字典树class，包含字典树的根节点root，和插入单词的方法insert
    class Trie {
        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        // 插入单词
        public int insert(String word) {
            TrieNode node = root;
            boolean isNew = false;
            // 倒序插入每个字母，
            for (int i = word.length() - 1; i >= 0; i--) {
                int c = word.charAt(i) - 'a';
                // 如果当前孩子不存在，就新建TrieNode，此时说明是新创建的路径
                if (node.children[c] == null) {
                    isNew = true;
                    node.children[c] = new TrieNode();
                }
                node = node.children[c];
            }
            // 如果不是新创建的单词，创建长度返回0
            return isNew ? word.length() + 1 : 0;
        }
    }

    // 820. 单词的压缩编码
    // 字典树 Trie
    public int minimumLengthEncoding(String[] words) {
        // 将word按照长度降序排序，这样能先创建更长的单词
        sort(words, (o1, o2) -> o2.length() - o1.length());
        Trie trie = new Trie();
        int len = 0;
        for (String word : words) {
            len += trie.insert(word);
        }
        return len;
    }

    // 5369. 统计作战单位数 时间复杂度 O(n^2)
    public int numTeams(int[] rating) {
        if (rating.length < 3)
            return 0;
        int n = rating.length;
        // 统计左边小于当前值的数目，大于当前值的数目
        int[][] left = new int[n][2];
        // 统计左边小于当前值的数目，大于当前值的数目
        int[][] right = new int[n][2];

        for (int i = 1; i < n - 1; i++) {
            int k = i - 1;
            while (k >= 0) {
                if (rating[k] < rating[i])
                    left[i][0]++;
                else left[i][1]++;
                k--;
            }
        }
        for (int i = n - 2; i >= 1; i--) {
            int k = i + 1;
            while (k < n) {
                if (rating[k] < rating[i])
                    right[i][0]++;
                else right[i][1]++;
                k++;
            }
        }
        int count = 0;
        for (int i = 1; i < n - 1; i++) {
            count += left[i][0] * right[i][1];
            count += left[i][1] * right[i][0];
        }
        return count;
    }

    // 面试题62. 圆圈中最后剩下的数字
    // 数学公式，反推，最终剩下的数字下标为0，每一轮最终ans的下标都会前移m个位置
    public int lastRemaining(int n, int m) {
        int ans = 0;
        for (int i = 2; i <= n; i++) {
            ans = (ans + m) % i;
        }
        return ans;
    }

    // 912. 排序数组, 计数排序
    public int[] sortArray(int[] nums) {
        int max = -50001, min = 50001;
        // 查找最大值和最小值
        for (int num : nums) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }
        int[] count = new int[max - min + 2];
        for (int num : nums) {
            count[num - min]++;
        }
        int idx = 0;
        for (int num = min; num <= max; num++) {
            int con = count[num - min];
            while (con-- > 0) {
                nums[idx++] = num;
            }
        }
        return nums;
    }

    // 581. 最短无序连续子数组 双指针法
    // 升序数组中，每个数字，不低于左边最大值，不大于右边最小值，以此为原则，用双指针法从左右同时开始遍历
    // 找到最左边不满足这个规则，和最右边不满足这个规则的数字。
    public int findUnsortedSubarray(int[] nums) {
        int len = nums.length;
        int[] leftMax = new int[len];
        int[] rightMin = new int[len];

        leftMax[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > leftMax[i - 1])
                leftMax[i] = nums[i];
            else leftMax[i] = leftMax[i - 1];
        }

        rightMin[len - 1] = nums[len - 1];
        for (int i = len - 2; i >= 0; i--) {
            if (nums[i] < rightMin[i + 1])
                rightMin[i] = nums[i];
            else rightMin[i] = rightMin[i + 1];
        }

        int left = 0, right = len - 1;
        while (left < right) {
            boolean fl = false, fr = false;
            if (leftMax[left] <= nums[left] && nums[left] <= rightMin[left]) {
                left++;
                fl = true;
            }
            if (leftMax[right] <= nums[right] && nums[right] <= rightMin[right]) {
                right--;
                fr = true;
            }
            if (!fl && !fr)
                break;
        }
        return right > left ? right - left + 1 : 0;

    }

    // 1111. 有效括号的嵌套深度
    // 尽可能减少AB两大阵营的深度差距，就把嵌套括号分给AB两组类似((,))
    // 而(), )(分到一个组，因为不会造成深度增加
    public int[] maxDepthAfterSplit(String seq) {
        int[] res = new int[seq.length()];
        for (int i = 1; i < seq.length(); i++) {
            if (seq.charAt(i) == seq.charAt(i - 1))
                res[i] = res[i - 1] ^ 1;
            else res[i] = res[i - 1];
        }
        return res;
    }

    // 1177. 构建回文串检测
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        List<Boolean> res = new ArrayList<>();
        // 之前超时，加上这段通过，注意，没必要保存1-len,2-len的长度，因为可以通过2-len,0-len相减得到
        // 统计从0-len每个段的26个字母出现情况
        int[][] cnt = new int[s.length() + 1][26];
        for (int i = 1; i <= s.length(); i++) {
            cnt[i] = copyOf(cnt[i - 1], 26);
            cnt[i][s.charAt(i - 1) - 'a']++;
        }

        for (int[] query : queries) {
            int left = query[0], right = query[1], k = query[2];
            int len = right - left + 1;
            // 如果长度为1，或者可以替换的字母不少于长度的一半
            if (left == right || k >= len / 2) {
                res.add(true);
                continue;
            }
            // 统计出现频率为奇数次的数量odd
            int odd = 0;
            for (int i = 0; i < 26; i++) {
                odd += (cnt[right + 1][i] - cnt[left][i]) % 2;
            }
            // 长度为奇数，去掉中间字母
            if (len % 2 > 0)
                odd--;
            // System.out.println(len+" "+odd+" "+k);
            res.add(odd <= 2 * k);
        }
        return res;
    }

    // 289. 生命游戏 位运算来保存状态信息
    // 活着活着死亡都只占1位，而int型为8位，用第二位来保存下一时刻的存活与否，然后将第二位的状态替代第一位的状态
    public void gameOfLife(int[][] board) {
        int m = board.length, n = board[0].length;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                int cnt = 0;
                for (int p = i - 1; p <= i + 1; p++)
                    for (int q = j - 1; q <= j + 1; q++) {
                        // 统计i,j周围八个细胞的状态，注意只取最后一位
                        if (!(p == i && q == j) && p >= 0 && p < m && q >= 0 && q < n) {
                            cnt += board[p][q] & 1;
                        }
                    }
                // 当前为活细胞
                if ((board[i][j] & 1) > 0) {
                    // 下一时刻还是存活
                    if (cnt == 2 || cnt == 3) {
                        board[i][j] = 0b11;
                    }
                    // 其它情况下下一时刻死亡，board[i][j]=0b01，不用变
                } else {
                    // 死细胞，能复活
                    if (cnt == 3)
                        board[i][j] = 0b10;
                    // 下一时刻还是死细胞，即为当前状态0b00,不用变
                }
            }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                // 右移一位，保留第二位的状态
                board[i][j] = board[i][j] >> 1;
        return;
    }

    // 8. 字符串转换整数 (atoi),使用有限状态自动机
    public int myAtoi(String str) {
        int sign = 1;
        int res = 0;
        Map<String, String[]> dfa = new HashMap<>();
        dfa.put("start", new String[]{"start", "signed", "in_number", "end"});
        dfa.put("signed", new String[]{"end", "end", "in_number", "end"});
        dfa.put("number", new String[]{"end", "end", "in_number", "end"});
        dfa.put("end", new String[]{"end", "end", "end", "end", "end"});

        String state = "start";
        for (int i = 0; i < str.length(); i++) {
            String move = dfa.get(state)[getCol(str.charAt(i))];
            if (move.equals("end"))
                break;
            int x = str.charAt(i) - '0';
            if (move.equals("in_number")) {
                if (sign == 1 && res > (Integer.MAX_VALUE - x) / 10)
                    return Integer.MAX_VALUE;
                if (sign == -1 && sign * res < (Integer.MIN_VALUE + x) / 10)
                    return Integer.MIN_VALUE;
                res = res * 10 + x;
                state = "number";
            }
            if (move.equals("signed")) {
                sign = str.charAt(i) == '+' ? 1 : -1;
                state = "signed";
            }
        }
        return sign * res;
    }

    public int getCol(char c) {
        if (c == ' ')
            return 0;
        if (c == '+' || c == '-')
            return 1;
        if (Character.isDigit(c))
            return 2;
        return 3;
    }

    // 面试题56 - II. 数组中数字出现的次数 II 位运算，状态自动机，具体见题解
    // https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/solution/cong-zhu-wei-tong-ji-dao-luo-ji-biao-da-shi-fang-s/
    public int singleNum(int[] nums) {
        int a = 0, b = 0;
        //a为对应位的1出现2次的记录，b为对应位出现1次的记录，ab共同组成该位出现的次数
        for (int num : nums) {
            b = ~a & (b ^ num);
            a = ~b & (a ^ num);
            System.out.println("a=" + a + ",b=" + b);
        }
        return b;
    }

    public String plusOne(String s) {
        int len = s.length();
        if (s.charAt(len - 1) == '0') {
            s = s.substring(len - 1) + '1';
            return s;
        }
        int c = 1;
        StringBuffer sb = new StringBuffer();
        int i = len - 1;
        for (; i >= 0; i--) {
            if (c == 0) break;
            int tmp = s.charAt(i) - '0' + c;
            c = tmp / 2;
            sb.append(tmp % 2);
        }
        if (c == 1)
            sb.append(1);
        String res = "";
        if (i >= 0)
            res = s.substring(0, i + 1);
        return res + sb.reverse().toString();
    }

    // 5377. 将二进制表示减到 1 的步骤数
    public int numSteps(String s) {
        int b = 0;
        int len = s.length();
        int end = s.length() - 1;
        StringBuffer sb = new StringBuffer();
        while (end > 0) {
            if (s.charAt(end) == '0') {
                end--;
                b++;
            } else {
                s = plusOne(s.substring(0, end + 1));
                end = s.length() - 1;
                b++;
            }
        }
        return b;
    }

    int maxSeqSum = 0;
    List<Integer> seq = new ArrayList<>();
    boolean[] bestSelect = new boolean[0];

    public void subSequence(int[] nums, int sum, int k, int csum, int n, boolean[] select) {
        if (n == k) {
            if (csum > maxSeqSum && csum > sum - csum) {
                bestSelect = select.clone();
                maxSeqSum = csum;
            }
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (n < k && !select[i]) {
                select[i] = true;
                seq.add(nums[i]);

                subSequence(nums, sum, k, csum + nums[i], n + 1, select);

                seq.remove(seq.size() - 1);
                select[i] = false;
            }
        }
    }

    // 5195. 最长快乐字符串 贪心思想，每次取数量最多的字母，如果这个字母不和前两个形成'aaa' 'bbb' 'ccc'的情况。
    public String longestDiverseString(int a, int b, int c) {
        MyChar[] myChar = new MyChar[]{new MyChar(a, 'a'), new MyChar(b, 'b'),
                new MyChar(c, 'c')};
        StringBuffer sb = new StringBuffer();
        while (true) {
            sort(myChar);
            if (sb.length() >= 2 && (sb.charAt(sb.length() - 1) == sb.charAt(sb.length() - 2)
                    && sb.charAt(sb.length() - 1) == myChar[2].c)) {
                if (myChar[1].num-- > 0) {
                    sb.append(myChar[1].c);
                } else break;
            } else {
                if (myChar[2].num-- > 0) {
                    sb.append(myChar[2].c);
                } else break;
            }
        }
        return sb.toString();
    }

    // 5376. 非递增顺序的最小子序列
    // 求一个子序列，其元素之和大于未包含在该子序列中的元素之和。元素顺序可调换。返回答案要求按照非降序排列
    // 要求返回长度最小，元素之和最大的子序列，注意到题干说了满足所有这些约束条件的解决方案是唯一的。
    // 所以可以先降序排序，然后从最大元素往后加，加到满足要求条件，一定是唯一的解决方案。
    public List<Integer> minSubsequence(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++)
            sum += nums[i];
        sort(nums);
        List<Integer> res = new ArrayList<>();
        int subSum = 0;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (subSum > sum - subSum) {
                return res;
            }
            res.add(nums[i]);
            subSum += nums[i];
        }
        return res;
    }

    // 72. 编辑距离 动态规划
    // 状态定义：dp[i][j]表示word1的前i个字符到word2的前j个字符的最少操作数
    // 状态转移方程：这是关键，三种操作
    // dp[i][j] = dp[i-1][j]+1,表示word1的前i-1个字符到word2的前j个字符，然后A末尾插入第i个字符
    // dp[i][j] = dp[i][j-1]+1,表示word1的前i个字符到word2的前j-1个字符，然后B末尾插入第j个字符
    // dp[i][j] = dp[i-1][j-1]+1,表示word1的前i-1个字符到word2的前j-1个字符，然后A末尾第i个字符替换成word2第j个字符，注意如果word1的第i字符和word2的第j字符相等，则不需要+1
    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        // 边界条件，如果有word1，word2中有字符串长度为0，则编辑距离等于另一个长度不为0的字符串的长度。
        for (int i = 0; i <= len1; i++)
            dp[i][0] = i;
        for (int j = 0; j <= len2; j++)
            dp[0][j] = j;
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                // 根据上面的几种情况讨论，写出转移方程
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1] - 1) + 1;
                }
            }
        }
        return dp[len1][len2];
    }

    // leetcode 生成括号，回溯算法解决
    List<String> stringList = new ArrayList<>();

    public void generateParenthesis(int n, String s, int left, int right) {
        if (s.length() == 2 * n) {
            stringList.add(new String(s));
            return;
        }
        if (left < n)
            generateParenthesis(n, s + "(", left + 1, right);
        if (left > right)
            generateParenthesis(n, s + ")", left, right + 1);
    }

    public List<String> generateParenthesis(int n) {
        stringList = new ArrayList<>();
        generateParenthesis(n, "", 0, 0);
        return stringList;
    }

    // 获取线段左右端点
    public int[] getStartEnd(int[] start, int[] end) {
        int[] res = new int[4];
        if (start[0] < end[0]) {
            res[0] = start[0];
            res[1] = start[1];
            res[2] = end[0];
            res[3] = end[1];
        } else {
            res[0] = end[0];
            res[1] = end[1];
            res[2] = start[0];
            res[3] = start[1];
        }
        return res;
    }

    // 面试题 16.03. 交点
    // 执行用时: 0 ms , 在所有Java提交中击败了100.00%的用户 内存消耗: 38.1 MB , 在所有Java提交中击败了100.00% 的用户
    public double[] intersection(int[] start1, int[] end1, int[] start2, int[] end2) {
        double[] res = new double[0];
        // 斜率不存在时
        if (start1[0] == end1[0] || start2[0] == end2[0]) {
            // line1和line2斜率都不存在
            if (start1[0] == end1[0] && start2[0] == end2[0]) {
                if (start1[0] != start2[0])  // 横坐标不等，不存在交点
                    return new double[0];
                int line1_maxY = Math.max(start1[1], end1[1]);
                int line1_minY = Math.min(start1[1], end1[1]);
                int line2_maxY = Math.max(start2[1], end2[1]);
                int line2_minY = Math.min(start2[1], end2[1]);
                // 横坐标相等，不存在交点
                if (line1_maxY < line2_minY || line1_minY > line2_maxY)
                    return res;
                else {
                    // 横坐标相等，存在交点，交点为两条线段各自y值最小值的较大值。
                    res = new double[]{(double) start1[0], (double) Math.max(line1_minY, line2_minY)};
                    return res;
                }
            } else {
                // 一条存在斜率，一条不存在斜率，如果是line2不存在斜率，交换line1和line2
                // 转换为 line1不存在斜率，line2存在斜率的情况。
                int[] tmp = new int[2];
                if (start2[0] == end2[0]) {
                    tmp = start1.clone();
                    start1 = start2.clone();
                    start2 = tmp;
                    tmp = end1.clone();
                    end1 = end2.clone();
                    end2 = tmp;
                }
                int x0, y0, x1, y1, x;
                x = start1[0];
                int[] t = getStartEnd(start2, end2);
                x0 = t[0];
                y0 = t[1];
                x1 = t[2];
                y1 = t[3];
                // 不存在交点
                if (x0 > x || x1 < x)
                    return res;
                else {
                    double k = (double) (y1 - y0) / (x1 - x0);
                    double y = y0 + k * (x - x0);
                    if (y > Math.max(start1[1], end1[1]) || y < Math.max(start1[1], end1[1]))
                        return res;
                    else {
                        res = new double[]{x, y};
                        return res;
                    }
                }
            }
        } else {
            double x0, y0, x1, y1, m0, n0, m1, n1;
            int[] t = getStartEnd(start1, end1);
            x0 = t[0];
            y0 = t[1];
            x1 = t[2];
            y1 = t[3];
            t = getStartEnd(start2, end2);
            m0 = t[0];
            n0 = t[1];
            m1 = t[2];
            n1 = t[3];
            double k1 = (double) (y1 - y0) / (x1 - x0), k2 = (double) (n1 - n0) / (m1 - m0);
            double b1 = y0 - k1 * x0, b2 = n0 - k2 * m0;
            if (Math.abs(k2 - k1) < 1e-6) { //斜率相同
                if (b1 - b2 > 1e-6) // y=kx+b,b不同，无交点
                    return res;
                else {
                    if (m0 > x1 || m1 < x0)
                        return res;
                    double x = Math.max(x0, m0);
                    double y = k1 * x + b1;
                    res = new double[]{x, y};
                    return res;
                }
            }
            // y0 + k1(x-x0) = n0 + k2(x-m0)
            // x = (double)(y0-n0 - (k1x0-k2m0) )/(k2-k1);
            double x = (double) (y0 - n0 - (k1 * x0 - k2 * m0)) / (k2 - k1);
            // 判断x0 <= x <= x1，注意相等时的判断，注意也要满足m0 <= x <= n0
            double maxX0 = Math.max(x0, m0), minX1 = Math.min(x1, m1);
            if ((x > maxX0 || Math.abs(x - maxX0) < 1e-6) && (x < minX1 || Math.abs(x - minX1) < 1e-6)) {
                res = new double[]{x, y0 + k1 * (x - x0)};
                return res;
            } else return res;
        }
    }

    // 01 矩阵
    public int[][] updateMatrix(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            return matrix;
        int M = matrix.length, N = matrix[0].length;
        boolean[][] flag = new boolean[M][N];

        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (matrix[i][j] == 0) {
                    q.offer(new int[]{i, j});
                    flag[i][j] = true;
                }
            }
        }
        int depth = -1;
        while (!q.isEmpty()) {
            int size = q.size();
            depth++;
            while (size-- > 0) {
                int[] s = q.poll();
                int r = s[0], c = s[1];
                matrix[r][c] = depth;
                if (r < M - 1 && !flag[r + 1][c]) {
                    q.offer(new int[]{r + 1, c});
                    flag[r + 1][c] = true;
                }
                if (r > 0 && !flag[r - 1][c]) {
                    q.offer(new int[]{r - 1, c});
                    flag[r - 1][c] = true;
                }
                if (c < N - 1 && !flag[r][c + 1]) {
                    q.offer(new int[]{r, c + 1});
                    flag[r][c + 1] = true;
                }
                if (c > 0 && !flag[r][c - 1]) {
                    q.offer(new int[]{r, c - 1});
                    flag[r][c - 1] = true;
                }
            }
        }
        return matrix;
    }

    // 56. 合并区间
    public int[][] merge(int[][] intervals) {
        if (intervals.length < 2)
            return intervals;
        sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        List<int[]> list = new ArrayList<>();
        int i = 1;
        while (i < intervals.length) {
            if (intervals[i][0] <= intervals[i - 1][1]) {
                intervals[i][0] = intervals[i - 1][0];
                intervals[i][1] = Math.max(intervals[i - 1][1], intervals[i][1]);
            } else {
                list.add(intervals[i - 1]);
            }
            if (i == intervals.length - 1)
                list.add(intervals[i]);
            i++;
        }
        int[][] res = new int[list.size()][2];
        for (i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public List<List<String>> displayTable(List<List<String>> orders) {
        Set<String> tables = new HashSet<>();
        Set<String> foods = new HashSet<>();
        // table hashmap
        Map<String, HashMap<String, Integer>> displays = new HashMap<>();
        for (int i = 0; i < orders.size(); i++) {
            List<String> order = orders.get(i);
            String table = order.get(1), food = order.get(2);
            HashMap<String, Integer> map = new HashMap<>();
            if (!tables.contains(table)) {
                tables.add(table);
                map.put(food, 1);
            } else {
                map = displays.get(table);
                int oldNum = map.getOrDefault(food, 0);
                map.put(food, oldNum + 1);
            }
            displays.put(table, map);
            foods.add(food);
        }

        Object[] tableArrays = tables.toArray();
        Arrays.sort(tableArrays);
        Object[] foodArrays = foods.toArray();
        Arrays.sort(foodArrays);
        List<List<String>> res = new ArrayList<>();

        List<String> header = new ArrayList<>();
        header.add("Table");
        int tableNum = tableArrays.length;
        int foodNum = foodArrays.length;
        for (int i = 0; i < foodNum; i++)
            header.add(foodArrays[i].toString());
        res.add(header);

        List<String> row = new ArrayList<>();
        for (Object tableArray : tableArrays) {
            row = new ArrayList<>();
            row.add(tableArray.toString());
            for (Object foodArray : foodArrays)
                row.add(displays.get(tableArray.toString()).get(foodArray.toString()).toString());
            res.add(row);
        }
        return res;
    }

    // 1248. 统计「优美子数组」
    // 思路:(上一个奇数位置 到 当前奇数位置的偶数个数) X (当前位置+k-1个奇数 到 其下一个奇数之间的偶数个数) = 子数组个数.
    // 挺拗口的,想明白就简单了.
    // 为了处理 +1 / -1的边界问题,添加2个哨兵元素[1] 到两边.这是一种常规处理方法~套用就好~
    public int numberOfSubarrays(int[] nums, int k) {
        List<Integer> odds = new ArrayList<>();
        odds.add(0);
        for (int i = 0; i < nums.length; i++) {
            if ((nums[i] & 1) > 0)
                odds.add(i + 1);
        }
        odds.add(nums.length + 1);
        int res = 0;
        for (int i = 1; i + k < odds.size(); i++) {
            res += (odds.get(i) - odds.get(i - 1)) * (odds.get(i + k) - odds.get(i + k - 1));
        }
        return res;
    }


    // 面试题 08.11. 硬币
    // 公式法求解 n分 x = 25a + 10b + 5c + d
    // a 最小取0，最大取[x/25]的向下取整记作a0。
    //对于任意a ∈ 0 <= a <= a0, 则 剩下的钱数为 x-25*a = 10b + 5c + d
    // 那么b的范围最小取0，最大取 [(x-25a)/10]向下取整。记作b0。
    // 对于任意的b ∈ 0 <= b <= b0, 则未取的b0-b个10分硬币，可以用2b0-2b个5分硬币替代。
    // 原来的c最大值为 (x-25a-10b0)/5向下取整,记作c0;
    // 则 剩余钱数为 x-25a-10b = 5(c0+2b0-2b)+ d
    // 所以5分硬币的取值范围为[0, c0+2b0-2b],一旦5分硬币数量确定，那么1分硬币的数量d自然就确定了。
    // 也即对于取a个25分硬币，b个10分硬币，表示x分硬币的方案有c0+2b0-2b+1种(0也是一种）。
    // 对于固定的a，对b利用等差数列公式，得到表示x分硬币的方案共有
    //             Σ(b=0, b0)(c0+2b0-2b+1) = (b0+1)(c0+b0+1)
    // 那么对所有a进行求和就能得到所有的表示方案。
    public int waysToChange(int n) {
        int mod = 1000000007;
        int a0 = n / 25;
        long res = 0;
        for (int a = 0; a <= a0; a++) {
            int rest = n - 25 * a;
            int b0 = rest / 10;
            int c0 = rest % 10 / 5;
            res = (res + (long) (b0 + 1) * (c0 + b0 + 1) % mod) % mod;
        }
        return (int) res;
    }

    int reverseNum = 0;

    // 面试题51. 数组中的逆序对
    public void mergeSort(int[] nums, int l, int r) {
        if (l == r)
            return;
        int mid = l + (r - l) / 2;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid + 1, r);
        merge(nums, l, mid, r);
    }

    public void merge(int[] nums, int left, int mid, int right) {
        int[] s1 = Arrays.copyOfRange(nums, left, mid + 1);
        int[] s2 = Arrays.copyOfRange(nums, mid + 1, right + 1);
        int i = 0, j = 0, k = left;
        while (i < s1.length || j < s2.length) {
            if (j == s2.length)
                nums[k++] = s1[i++];
            else if (i == s1.length)
                nums[k++] = s2[j++];
            else if (s1[i] > s2[j]) {
                nums[k++] = s2[j++];
                reverseNum += s1.length - i;
            } else nums[k++] = s1[i++];
        }
    }

    public int reversePairs(int[] nums) {
        reverseNum = 0;
        if (nums.length < 2)
            return reverseNum;
        mergeSort(nums, 0, nums.length - 1);
        return reverseNum;
    }

    // 33. 搜索旋转排序数组
//    public int search(int[] nums, int target) {
//        if (nums.length == 0)
//            return -1;
//        int len = nums.length;
//        int left = 0, right = len - 1;
//        int mid;
//        while(left <= right){
//            mid = left + (right-left)/2;
//            // System.out.println(mid);
//            if (nums[mid] == target)
//                return mid;
//            // order为1，左边有序，order为2，右边有序
//            int order = 1;
//            // 左边有序 4 5 6 7 0 1 2 mid在6处
//            if (nums[mid] > nums[right]){
//                if (target < nums[mid]){
//                    if (target >= nums[left])
//                        right = mid - 1;
//                    else
//                        left = mid + 1;
//                }else left = mid + 1;
//            }else{
//                if (target < nums[mid])
//                    right = mid-1;
//                else{
//                    if (target <= nums[right])
//                        left = mid + 1;
//                    else
//                        right = mid - 1;
//                }
//            }
//        }
//        return -1;
//    }

    // 面试题56 - I. 数组中数字出现的次数 异或运算
    public int[] singleNumbers(int[] nums) {
        int n = 0;
        for (int num : nums)
            n ^= num;
        int p = 1;
        while ((p & n) == 0) {
            p <<= 1;
        }

        int a = 0, b = 0;
        for (int num : nums) {
            if ((num & p) != 0)
                a ^= num;
            else
                b ^= num;
        }
        return new int[]{a, b};
    }

    // 1095. 山脉数组中查找目标值
    // 先二分搜索查找到山峰位置，然后再左右两边各进行一次二分搜索
    // 注意山峰位置，在左右搜索时，只在某一边搜索一次即可，也即搜索区间[0,peak], [peak+1,length-1]
    // 二分搜索时，left和right不要取等号，和mid处value进行大小比较时，left取mid+1，right取mid，就不会陷入死循环。
    public int findInMountainArray(int target, MountainArray mountainArr) {
        int peakPos = -1;
        int left = 0, right = mountainArr.length() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            // System.out.println(left+","+right+","+mid);
            int midNum = mountainArr.get(mid);
            int leftNum = mountainArr.get(mid - 1), rightNum = mountainArr.get(mid + 1);
            if (midNum > leftNum && midNum > rightNum) {
                if (target == midNum)
                    return mid;
                if (target == leftNum)
                    return mid - 1;
                peakPos = mid;
                break;
            } else if (leftNum < midNum) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // 查找左边
        left = 0;
        right = peakPos;
        while (left < right) {
            int mid = left + (right - left) / 2;
            int midNum = mountainArr.get(mid);
            if (midNum == target)
                return mid;
            if (midNum < target)
                left = mid + 1;
            else
                right = mid;
        }
        // 注意有可能右边区间只有一个数，单独判断
        if (mountainArr.get(left) == target)
            return left;
        // 查找右边
        left = peakPos + 1;
        right = mountainArr.length() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            // System.out.println(left+","+right+","+mid);
            int midNum = mountainArr.get(mid);
            if (midNum == target)
                return mid;
            if (midNum > target)
                left = mid + 1;
            else
                right = mid;
        }
        // 注意有可能右边区间只有一个数，单独判断
        if (mountainArr.get(left) == target)
            return left;
        return -1;
    }

    // 无重复字符的最长子串,动态规划解法
    public int lengthOfLongestSubstring(String s) {
        if (s.length() < 2)
            return s.length();
        int len = s.length();
        int[] length = new int[len];
        int[] last = new int[256];
        Arrays.fill(last, -1);
        length[0] = 1;
        last[s.charAt(0)] = 0;
        int maxLen = 1;
        for (int i = 1; i < len; i++) {
            if (s.charAt(i) == s.charAt(i - 1)) {
                length[i] = 1;
                last[s.charAt(i)] = i;
                continue;
            }
            int pos = last[s.charAt(i)];
            if (pos < i - length[i - 1]) {
                length[i] = length[i - 1] + 1;
            } else {
                length[i] = i - pos;
            }
            last[s.charAt(i)] = i;
            maxLen = Math.max(maxLen, length[i]);
        }
        return maxLen;
    }

    // 983. 最低票价 动态规划
    public int mincostTickets(int[] days, int[] costs) {
        int day = days[days.length - 1];
        int[] dp = new int[day + 1];
        int k = 0;
        int startDay = days[0];
        for (int i = startDay; i <= day; i++) {
            if (i < days[k]) { // 当天没有出游计划
                dp[i] = dp[i - 1];
                continue;
            }
            dp[i] = Math.min(dp[i - 1] + costs[0],
                    Math.min(dp[Math.max(i - 7, 0)] + costs[1], dp[Math.max(i - 30, 0)] + costs[2]));
            k++;
        }
        return dp[day];
    }

    // 53. 最大子序和,动态规划解法
//    public int maxSubArray(int[] nums) {
//        int maxSum = nums[0];
//        int sum = nums[0];
//        for (int i = 1; i < nums.length; i++){
//            if (sum < 0) // 如果前面的sum是负数，就不加
//                sum = nums[i];
//            else
//                sum += nums[i];
//            maxSum = Math.max(maxSum, sum);
//        }
//        return maxSum;
//    }

    // 53. 最大子序和，分治法
    // key points: 对[l, r]的子序列维护一个状态
    // 包含左端点开始的最大子序和lSum，右端点开始的最大子序和rSum，区间和iSum，区间内的最大子序和mSum
    public Status pushUp(Status l, Status r) {
        int lSum = Math.max(l.lSum, l.iSum + r.lSum);
        int rSum = Math.max(r.rSum, r.iSum + l.rSum);
        int iSum = l.iSum + r.iSum;
        int mSum = Math.max(Math.max(l.mSum, r.mSum), l.rSum + r.lSum);
        return new Status(lSum, rSum, mSum, iSum);
    }

    public Status get(int l, int r, int[] nums) {
        if (l == r) {
            return new Status(nums[l], nums[l], nums[l], nums[l]);
        }
        int m = l + (r - l) / 2;
        Status lStatus = get(l, m, nums);
        Status rStatus = get(m + 1, r, nums);
        return pushUp(lStatus, rStatus);
    }

    public int maxSubArray(int[] nums) {
        return get(0, nums.length - 1, nums).mSum;
    }

    // 560. 和为K的子数组 前缀和+优化
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> sumFreq = new HashMap<>();
        sumFreq.put(0, 1);
        int preSum = 0;
        int res = 0;
        for (int num : nums) {
            preSum += num;
            if (sumFreq.containsKey(preSum - k))
                res += sumFreq.get(preSum - k);
            sumFreq.put(preSum, sumFreq.getOrDefault(preSum, 0) + 1);
        }
        return res;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses == 0)
            return new int[0];
        int[][] edge = new int[numCourses][numCourses];
        int[] in = new int[numCourses];
        // 建立邻接矩阵，更新入度列表
        for (int[] pre : prerequisites) {
            edge[pre[1]][pre[0]] = 1;
            in[pre[0]]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (in[i] == 0)
                q.offer(i);
        }
        int[] res = new int[numCourses];
        int k = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                int n = q.poll();
                res[k++] = n;
                for (int i = 0; i < numCourses; i++) {
                    if (i != n && edge[n][i] == 1) {
                        in[i]--;
                        if (in[i] == 0)
                            q.offer(i);
                    }
                }
            }
        }
        if (k == numCourses)
            return res;
        else
            return new int[0];
    }


    public static void main(String[] args) {
//        int[] a = {1, 2, 3, 4, 5, 6, 7};
//        int k = 3;
//        rotate1(a, k);
//        for (int i=0;i<a.length;i++){
//            System.out.println(a[i]);
//        }

//        int a = 1534236469;
//        System.out.println(reverse(a));

//        String s = "zdz"
//        System.out.println(alphabetBoardPath(s));

//        int[] a = {9, 2, 9};
//        int[] b = plusOne(a);
//        for(int i=0;i<b.length;i++)
//            System.out.println(b[i]);

//        String s = "A man, a plan, a canal: Panama";
//        System.out.println(isPalindrome(s));

//        int[] nums1 = new int[]{1, 2};
//        int[] nums2 = new int[]{3, 4};
//       System.out.println(findMedianSortedArrays(nums1, nums2));
//
//        int[] nums = new int[]{7, 1, 5, 3, 6, 4};
//        System.out.println(maxProfit(nums));

//        String str = "42";
//        System.out.println(myAtoi(str));

//        String s = "mississippi", t = "issip";
//        System.out.println(strStr(s, t));

//        int n = 6;
//        System.out.println(countAndSay(n));

//        String[] strs = new String[]{"flower","flow","flight"};
////        System.out.println(longestCommonPrefix(strs));

//        String[] queries = {"bbb","cc"}, words = {"a","aa","aaa","aaaa"};
//        int[] answer = numSmallerByFrequency(queries, words);
//        for (int i=0;i<answer.length;i++)
//            System.out.println(answer[i]);

//        System.out.println(numPrimeArrangements(100));

//        int[] calories = new int[]{3, 2};
//        System.out.println(dietPlanPerformance(calories,2,0,1));

//        String[] words = new String[]{"aaaa", "asas", "able", "ability", "actt", "actor", "access"};
//        String[] puzzles = new String[]{"aboveyz", "abrodyz", "abslute", "absoryz", "actresz", "gaswxyz"};
//        List<Integer> nums = findNumOfValidWords(words, puzzles);
//        System.out.println(nums.toString());

//        int[] nums1 = {0}

//        int[] nums2 = {1};
//        int m = 0, n = 1;
//        merge(nums1, m, nums2, n);
//        for (int i=0; i< nums1.length; i++){
//            System.out.println(nums1[i]);
//        }

//        System.out.println(longestPalindrome("aaa"));

//        int[] nums = {-2,0,1,1,2};
//        List<List<Integer>> res = threeSum(nums);

//        System.out.println(countPrimes(20));

//        System.out.println(isPowerOfThree(242));

//        System.out.println(romanToInt("DCXXI"));

//        System.out.println(hammingDistance(3, 1));

//        System.out.println(reverseBits(429496794));

//        System.out.println(my_abs(-666));

//    System.out.println(lengthOfLongestSubstring("dvdf"));

//        System.out.println(simplifyPath("/a//b////c/d//././/.."));

//        System.out.println(restoreIpAddresses("25525511135"));

//        int[] nums = {1};
//        System.out.println(permute(nums));
//        int[] nums = {1, 1, 1, 1, 1};
//        System.out.println(increasingTriplet(nums));

//        char[][] grid = {{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}};
//        System.out.println(numIslands(grid));


        Main main = new Main();
//        System.out.println(main.divide(-2147483648, 2));
//        System.out.println(main.trailingZeroes(200));
//        System.out.println(main.fractionToDecimal(1, 33));
//        System.out.println(main.evalRPN(new String[]{"10","6","9","3","+","-11","*","/","*","17","+","5","+"}));
//        System.out.println(main.majorityElement(new int[]{6, 5, 5}));
//        System.out.println(main.checkInclusion("adc", "dcda"));

//        int[][] grid = {{0,0,1,0,0,0,0,1,0,0,0,0,0}, {0,0,0,0,0,0,0,1,1,1,0,0,0},{0,1,1,0,1,0,0,0,0,0,0,0,0},{0,1,0,0,1,1,0,0,1,0,1,0,0}, {0,1,0,0,1,1,0,0,1,1,1,0,0}, {0,0,0,0,0,0,0,0,0,0,1,0,0}, {0,0,0,0,0,0,0,1,1,1,0,0,0}, {0,0,0,0,0,0,0,1,1,0,0,0,0}};
//        System.out.println(main.maxAreaOfIsland(grid));
//        int[][] M = {{1,0,0,1}, {0,1,1,0}, {0,1,1,1}, {1,0,1,1}};
//        System.out.println(main.findCircleNum(M));
//            System.out.println(main.convert("LEETCODEISHIRING", 4));
//        System.out.println(main.getPermutation(3, 3));
//        System.out.println(Math.ceil(5/3.));

//        int[] houses = {1, 2, 3};
//        int[] heaters = {1, 2, 3};
//        System.out.println(main.findRadius(houses, heaters));  // 161834419
//        int[][] envelopes = {{15,8}, {2,20}, {2,14}, {4,17}, {8,19}, {8,9}, {5,7}, {11,19}, {8,11}, {13,11}, {2,13}, {11,19}, {8,11}, {13,11}, {2,13}, {11,19}, {16,1}, {18,13}, {14,17}, {18,19}};
//        System.out.println(main.maxEnvelopes(envelopes));

//        char[][] board = {{'O','X','X','O','X'},{'X','O','O','X','O'},{'X','O','X','O','X'},{'O','X','O','O','O'},{'X','X','O','X','O'}};
//        main.solve(board);

//        String[] words = {"Hello","Alaska","Dad","Peace"};
//        main.findWords(words);
//        int[] gas = {5, 1, 2, 3, 4};
//        int[] cost = {4, 4, 1, 5, 1};
//        System.out.println(main.canCompleteCircuit(gas, cost));
//        System.out.println(main.numSquares(48));
//        main.partition("efe");
//        System.out.println(main.largestNumber(new int[]{3, 30, 34, 5, 9}));
//        System.out.println(main.calculate("1*2-3/4+5*6-7*8+9/10") );
//        System.out.println(main.longestSubstring("zzzzzzzzzzaaaaaaaaabbbbbbbbhbhbhbhbhbhbhicbcbcibcbccccccccccbbbbbbbbaaaaaaaaafffaahhhhhiaahiiiiiiiiifeeeeeeeeee", 10));
//        System.out.println(main.orangesRotting(new int[][]{{0, 2, 2}}));
//        System.out.println(main.sumFourDivisors(new int[]{7286,18704,70773,8224,91675}));
//        System.out.println(main.longestPrefix("level"));
//        System.out.println(main.hasGroupsSizeX(new int[]{1,1,2,2,2,2}));
//        System.out.println(main.minIncrementForUnique(new int[]{3, 2, 1, 2, 1, 7}));
//        System.out.println(main.minimumLengthEncoding(new String[]{"time", "me", "bell"}));
//        System.out.println(main.numTeams(new int[]{1,2,3,4}));
//        System.out.println(main.singleNum(new int[]{3, 4, 3, 3}));
//        System.out.println(main.numSteps("1"));
//        System.out.println(main.minSubsequence(new int[]{73,71,26,65,53,63,46,54,45,1,69,70,80,79,43,59,77,4,30,82,23,71,75,24,12,71,33,5,87,10,11,3,58,2,97,97,36,32,35,15,80,24,45,38,9,22,21,33,68,22,85,35,83,92,38,59}));
//        System.out.println(main.longestDiverseString(7, 1, 0));
//        main.intersection(new int[]{12,-55}, new int[]{59,-60}, new int[]{4,-55}, new int[]{81,-62});
//        List<List<String>> displays = new ArrayList<>();
//        displays.add(asList( new String[]{"David", "3", "Ceviche"}));
//        displays.add(asList(new String[]{"Corina","10","Beef Burrito"}));
//        displays.add(asList(new String[]{"David","3","Fried Chicken"}));
//        displays.add(asList(new String[]{"Carla","5","Water"}));
//        displays.add(asList(new String[]{"Carla","5","Ceviche"}));
//        displays.add(asList(new String[]{"Rous","3","Ceviche"}));
//        main.displayTable(displays);
//        System.out.println(main.numberOfSubarrays(new int[]{2,2,2,1,2,2,1,2,2,2}, 2));
//        System.out.println(main.waysToChange(900750));
//        System.out.println(main.reversePairs(new int[]{7, 6, 5, 4}));
//        System.out.println(main.singleNum(new int[]{4,1,4,6}));
//        System.out.println(main.lengthOfLongestSubstring("pwwkew"));
        System.out.print(main.findOrder(2, new int[0][0]));
    }
}

class MyChar implements Comparable<MyChar> {
    public int num;
    public char c;

    public MyChar(int n, char xc) {
        num = n;
        c = xc;
    }

    @Override
    public int compareTo(MyChar o2) {
        return num - o2.num;
    }
}

// 1095. 山脉数组中查找目标值 山脉接口
interface MountainArray {
    public int get(int index);

    public int length();
}

class Status {
    int lSum, rSum, mSum, iSum;

    public Status(int l, int r, int m, int i) {
        lSum = l;
        rSum = r;
        mSum = m;
        iSum = i;
    }
}
