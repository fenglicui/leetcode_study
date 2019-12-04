/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: RandomizedSet
 * Author:   cfl
 * Date:     2019/11/7 14:49
 * Description: leetcode Insert Delete GetRandom O(1)
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * 〈用hashmap + arraylist实现〉<br>
 * 〈leetcode Insert Delete GetRandom O(1)〉
 * 执行用时：22 ms
 * 战胜99.68%的Java提交记录
 *
 * @author cfl
 * @create 2019/11/7
 * @since 1.0.0
 */
public class RandomizedSet {

    // 存储list中的元素对应的下标，以此达到O(1)时间复杂度
    private static HashMap<Integer, Integer> map = null;
    private static List<Integer> list = null;

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {
        map = new HashMap<>();
        list = new ArrayList<>();
    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified element.
     */
    public boolean insert(int val) {
        if (map.containsKey(val)) return false;
        list.add(val);
        map.put(val, list.size() - 1);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!map.containsKey(val)) return false;
        // 先在map中删除元素，然后删除list末尾元素，并将已删除元素的下标处的值改为末尾元素,然后更新map
        int valIndex = map.remove(val);
        int lastval = list.remove(list.size() - 1);
        // 删掉的不是末尾元素
        if (lastval != val) {
            list.set(valIndex, lastval);
            map.put(lastval, valIndex);
        }
        return true;
    }

    /**
     * Get a random element from the set.
     */
    // 随机生成要获取元素的下标索引
    public int getRandom() {
        int index = new Random().nextInt(list.size());
        return list.get(index);
    }

    public static void main(String[] args) {
        RandomizedSet randomizedSet = new RandomizedSet();
        System.out.println(randomizedSet.insert(5));
        System.out.println(randomizedSet.remove(2));
        System.out.println(randomizedSet.insert(2));
        System.out.println(randomizedSet.getRandom());
        System.out.println(randomizedSet.remove(5));
        System.out.println(randomizedSet.insert(2));
        System.out.println(randomizedSet.getRandom());
    }
}
