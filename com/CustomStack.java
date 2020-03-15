package com;

import java.util.ArrayList;
import java.util.List;

class CustomStack {
    // 用一个arrayList来维护
    private List<Integer> list = null;
    private int size;
    private int maxSize;

    public CustomStack(int maxSize) {
        list = new ArrayList<>();
        this.maxSize = maxSize;
        size = 0;
    }

    public void push(int x) {
        if (size < maxSize) {
            list.add(x);
            size++;
        }
    }

    public int pop() {
        if (size > 0) {
            int v = list.remove(size - 1);
            size--;
            return v;
        } else return -1;
    }

    public void increment(int k, int val) {
        if (size == 0)
            return;
        int n = Math.min(k, size);
        // 栈底的 k 个元素的值都增加 val
        for (int i = 0; i < n; i++) {
            list.set(i, list.get(i) + val);
        }
    }
}

/**
 * Your CustomStack object will be instantiated and called as such:
 * CustomStack obj = new CustomStack(maxSize);
 * obj.push(x);
 * int param_2 = obj.pop();
 * obj.increment(k,val);
 */
