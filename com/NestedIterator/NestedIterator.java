package com.NestedIterator;

import java.util.Iterator;

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 * <p>
 * // @return true if this NestedInteger holds a single integer, rather than a nested list.
 * public boolean isInteger();
 * <p>
 * // @return the single integer that this NestedInteger holds, if it holds a single integer
 * // Return null if this NestedInteger holds a nested list
 * public Integer getInteger();
 * <p>
 * // @return the nested list that this NestedInteger holds, if it holds a nested list
 * // Return null if this NestedInteger holds a single integer
 * public List<NestedInteger> getList();
 * }
 */
// 341. 扁平化嵌套列表迭代器，保存一个Iterator，用于检索嵌套列表
// 每次遇到一个嵌套列表，new一个Iterator去迭代
public class NestedIterator implements Iterator<Integer> {
    private int current = 0;
    private List<NestedInteger> list = null;
    private NestedIterator it = null;
    private int size = 0;

    public NestedIterator(List<NestedInteger> nestedList) {
        list = nestedList;
        size = list.size();
    }

    // 如果当前项的Iterator有next返回元素值，没有遍历下一个元素，进而判断其类型是Integer还是嵌套列表
    @Override
    public Integer next() {
        if (it != null) {
            // 当前it未遍历完
            if (it.hasNext()) {
                return it.next();
            } else {
                // 当前it正好遍历到最后一个元素，下标指针往后走一个
                current++;
                it = null;
            }
        }
        NestedInteger nested = list.get(current);
        // 如果下一个是Integer，下标+1，直接返回
        if (nested.isInteger()) {
            current++;
            return nested.getInteger();
        } else {
            // 否则新建一个List<NestedInteger>，继续迭代
            it = new NestedIterator(nested.getList());
            return next();
        }
    }

    // 如果当前项的Iterator有next返回true，没有遍历下一个元素，进而判断其类型是Integer还是嵌套列表,判断是否有next
    @Override
    public boolean hasNext() {
        if (current >= size)
            return false;
        if (it != null) {
            // 如果当前list有next，返回true
            if (it.hasNext())
                return true;
            else {
                // 否则往后寻找
                current++;
                it = null;
            }
        }
        if (current == size)
            return false;
        NestedInteger nested = list.get(current);
        // 下一个是integer，返回true
        if (nested.isInteger())
            return true;
        else {
            // 否则new一个Iterator，继续迭代
            it = new NestedIterator(nested.getList());
            return hasNext();
        }
    }
}

//预加载所有元素，dfs搜索
//public class NestedIterator implements Iterator<Integer> {
//    List<Integer> list = null;
//    int con = 0;
//    int size = 0;
//
//    public void getList(List<NestedInteger> nestedList){
//        for (NestedInteger nested: nestedList){
//            if (nested.isInteger())
//                list.add(nested.getInteger());
//            else{
//                List<NestedInteger> nestedlist = nested.getList();
//                getList(nestedlist);
//            }
//        }
//    }
//
//    public NestedIterator(List<NestedInteger> nestedList) {
//        list = new ArrayList<>();
//        getList(nestedList);
//        size = list.size();
//    }
//
//    @Override
//    public Integer next() {
//        return list.get(con++);
//    }
//
//    @Override
//    public boolean hasNext() {
//        return con < size;
//    }
//}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */
