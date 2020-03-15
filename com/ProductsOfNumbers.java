package com;

import java.util.ArrayList;
import java.util.List;

// leetcode medium 最后k个数的乘积
class ProductOfNumbers {

    private List<Integer> nums = null;

    public ProductOfNumbers() {
        nums = new ArrayList<>();
    }

    public void add(int num) {
        nums.add(num);
    }

    public int getProduct(int k) {
        int len = nums.size();
        int res = 1;
        for (int i = len - 1; i > len - 1 - k; i--)
            res *= nums.get(i);
        return res;
    }
}

/**
 * Your ProductOfNumbers object will be instantiated and called as such:
 * ProductOfNumbers obj = new ProductOfNumbers();
 * obj.add(num);
 * int param_2 = obj.getProduct(k);
 */