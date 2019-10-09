/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: Solution
 * Author:   cfl
 * Date:     2019/9/14 18:05
 * Description: shuffle an array
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com.shuffle_an_array;

import java.util.Random;

/**
 * 〈一句话功能简述〉<br>
 * 〈shuffle an array〉
 *
 * @author cfl
 * @create 2019/9/14
 * @since 1.0.0
 */
public class Solution {

    public int[] origin;

    public Solution(int[] nums) {
        this.origin = nums;
    }

    /**
     * Resets the array to its original configuration and return it.
     */
    public int[] reset() {
        return origin;
    }

    /**
     * Returns a random shuffling of the array.
     */
    public int[] shuffle() {
        Random random = new Random();
        int[] perm = origin.clone();
        int len = origin.length;
        // 每次生成一个数字，用作和当前位置元素交换数字的下标，然后二者交换
        for (int i = 0; i < len; i++) {
            int seq = random.nextInt(len);
            int tmp = perm[i];
            perm[i] = perm[seq];
            perm[seq] = tmp;
        }
        return perm;
    }
}
