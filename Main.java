import java.util.*;
import java.util.stream.Collectors;

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
        Arrays.sort(nums1);
        Arrays.sort(nums2);
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
        Arrays.sort(char_s);
        Arrays.sort(char_t);
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

    //leetcode 37 字符串转换整数（atoi）
    public static int myAtoi(String str) {
        if (str == null) return 0;
        //去除空格
        str = str.trim();
        if (str.length() == 0) return 0;
        char first = str.charAt(0);
        //第一个非空字符不是数字或正负号
        if (!Character.isDigit(first) && first != '+' && first != '-') return 0;
        int num = 0;
        int flag = 1;
        if (first == '-') flag = -1;
        else if (first != '+') num = first - '0';
        for (int i = 1; i < str.length(); i++) {
            char c = str.charAt(i);
            if (!Character.isDigit(c)) break;
            int tmp = c - '0';
            if (flag > 0 && num > (Integer.MAX_VALUE - tmp) / 10)
                return Integer.MAX_VALUE;
            if (flag < 0 && num * flag < (Integer.MIN_VALUE - flag * tmp) / 10)
                return Integer.MIN_VALUE;
            num = num * 10 + tmp;
        }
        return num * flag;
    }

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
        Arrays.sort(ints);
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
        Arrays.sort(word_ints);
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

    //Leetcode 构建回文串检测 超时
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        List<Boolean> nums = new ArrayList<>();
        for (int i = 0; i < queries.length; i++) {
            int left = queries[i][0];
            int right = queries[i][1];
            int k = queries[i][2];
            if (left == right) {
                nums.add(true);
                continue;
            }
            String substring = s.substring(left, right);
            char[] chars = substring.toCharArray();
            Arrays.sort(chars);
            int charcount = chars.length;
            for (int j = 0; j < chars.length - 1; j++) {
                if (chars[j] == chars[j + 1]) {
                    charcount -= 2;
                    j++;
                }
            }
            if (charcount == 0 || charcount / 2 <= k)
                nums.add(true);
            else nums.add(false);
//            System.out.println(substring);
        }
        return nums;
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
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2 && nums[i] <= 0; i++) {
            // 过滤连续重复的值  !!!重要！！！
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1, r = nums.length - 1;
            int sum = -nums[i];
            while (l < r) {
                // if (nums[r] + nums[r-1] < sum) break;
                if (nums[l] + nums[r] == sum) {
                    int[] a = {nums[i], nums[l], nums[r]};
                    List<Integer> tripe = Arrays.stream(a).boxed().collect(Collectors.toList());
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

    // leetcode 无重复字符的最长子串
    // 动态规划解法
    public static int lengthOfLongestSubstring(String s) {
        if (s.length() < 2)
            return s.length();
        int len = s.length();
        int maxsublen = 1;
        // 记录字符最后出现的下标
        Map<Integer, Integer> lastsub = new HashMap<>();
        // 记录以当前元素结尾的子串的最长子串长度
        int[] sublen = new int[len];
        lastsub.put((int) s.charAt(0), 0);
        sublen[0] = 1;
        for (int i = 1; i < len; i++) {
            int c = s.charAt(i);
            // 当前字母和上一个字母一样
            if (c == s.charAt(i - 1))
                sublen[i] = 1;
                // 当前字母和上一个字母不一样,且当前字母没出现过
            else if (!lastsub.containsKey(c))
                sublen[i] = sublen[i - 1] + 1;
                // 当前字母出现过，找到上一次出现的下标，看是否在以上个元素结尾的子串中
            else
                sublen[i] = Math.min(sublen[i - 1] + 1, i - lastsub.get(c));
            // 更新最后出现下标
            lastsub.put(c, i);
            maxsublen = sublen[i] > maxsublen ? sublen[i] : maxsublen;
        }
        return maxsublen;
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

//    // leetcode 全排列 回溯法求解
//    static List<List<Integer>> results = new ArrayList<>();
//    // 回溯算法 关键是for循环，回溯后记得更改访问状态，和排列对象，编写返回条件
//    // flag记录当前数字是否出现的标记，result为当前排列
//    public static void permute_backtrack(int[] nums, boolean[] flag, List<Integer> result){
//        if (result.size() == nums.length){
//            // 注意这里要new一个ArrayList，不然操作的是同一个List
//            results.add(new ArrayList<>(result));
//            return;
//        }
//        for (int i = 0; i < nums.length; i++){
//            if (flag[i])continue;
//            result.add(nums[i]);
//            flag[i] = true;
//            permute_backtrack(nums, flag, result);
//            flag[i] = false;
//            result.remove(result.size()-1);
//        }
//    }
//    // leetcode 全排列 回溯法求解
//    public static List<List<Integer>> permute(int[] nums) {
//        results = new ArrayList<>();
//        if (nums.length == 0)return results;
//        List<Integer> result = new ArrayList<>();
//        boolean[] flag = new boolean[nums.length];
//        permute_backtrack(nums,flag, result);
//        return results;
//    }

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
        List<Integer> collect = Arrays.stream(ints).boxed().collect(Collectors.toList());
        System.out.println(collect.size());

        // int[] -> Integer[]
        Integer[] integers = Arrays.stream(ints).boxed().toArray(Integer[]::new);

        //Integer[] -> int[]
        int[] ints1 = Arrays.stream(integers1).mapToInt(Integer::valueOf).toArray();

        //Integer[] -> list<Integer>
        List<Integer> integers2 = new ArrayList<>(Arrays.asList(integers1));
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


    // leetcode 生成括号
    // 回溯算法解决
    List<String> brackets = new ArrayList<>();

    //     回溯函数 s 当前括号串，l 左括号的个数，k当前总括号个数，n 括号对数
    public void back(String s, int l, int k, int n) {
        if (k == n * 2) {
            brackets.add(s);
            return;
        }
//         还有左括号可以用
        if (l < n)
            back(s + "(", l + 1, k + 1, n);
//         左括号多余右括号
        if (l > k - l)
            back(s + ")", l, k + 1, n);
    }

    public List<String> generateParenthesis(int n) {
        if (n == 0)
            return brackets;
        String s = "";
        back(s, 0, 0, n);
        return brackets;
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
            }
            else if (nums[i] == 1)
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
        Arrays.sort(nums);
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

        Arrays.sort(coins);
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
////        System.out.println(numIslands(grid));

        Main main = new Main();
        System.out.println(main.divide(-2147483648, 2));

    }
}
