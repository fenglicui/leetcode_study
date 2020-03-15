package com.LRU;

import java.util.HashMap;
import java.util.Map;


// leetcode 哈希表+双向链表 哈希链表 实现LRU缓存机制
// 自定义node，包含key，val，前驱和后继
class Node {
    int key, val;
    Node prev, next;

    public Node(int key, int val) {
        this.key = key;
        this.val = val;
        prev = null;
        next = null;
    }
}

// 双向链表结构，包括头尾指针和size
class DoubleList {
    Node head, tail;
    int size;

    public DoubleList() {
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    // 首端添加节点
    public void addFirst(Node node) {
        // node的prev和next
        node.next = head.next;
        node.prev = head;
        // 原来next和head
        head.next.prev = node;
        head.next = node;
        size++;
    }

    // 删除指定节点(一定存在)
    public void remove(Node node) {
        // 原next和prev
        node.next.prev = node.prev;
        node.prev.next = node.next;
        size--;
    }

    // 删除并返回末端节点
    public Node removeLast() {
        if (tail.prev == head)
            return null;
        Node node = tail.prev;
        remove(node);
        return node;
    }

    public int getSize() {
        return size;
    }
}

class LRUCache {
    // node <k,v>
    Map<Integer, Node> map;
    // <k, v> -> <k, v, prev, next>
    DoubleList doubleList;
    int capacity;

    public LRUCache(int capacity) {
        map = new HashMap<>();
        doubleList = new DoubleList();
        this.capacity = capacity;
    }

    public int get(int key) {
        if (!map.containsKey(key))
            return -1;
        Node node = map.get(key);
        // 用put将node提到前头
        put(node.key, node.val);
        return node.val;
    }

    public void put(int key, int value) {
        // 创建新节点
        Node node = new Node(key, value);
        // 如果包含key
        if (map.containsKey(key)) {
            // 删除原有节点
            doubleList.remove(map.get(key));
            // 在首端添加新节点
            doubleList.addFirst(node);
            // 更新map
            map.put(key, node);
        } else {
            // 缓存容量达到上限，删除末尾节点，修正map
            if (doubleList.getSize() == capacity) {
                Node last = doubleList.removeLast();
                map.remove(last.key);
            }
            // 直接添加到首端，更新map
            doubleList.addFirst(node);
            map.put(key, node);
        }
    }

    public static void main(String[] args) {
        LRUCache cache = new LRUCache(2 /* 缓存容量 */);

        cache.put(1, 1);
        cache.put(2, 2);
        System.out.println(cache.get(1));       // 返回  1
        cache.put(3, 3);    // 该操作会使得密钥 2 作废
        System.out.println(cache.get(2));       // 返回 -1 (未找到)
        cache.put(4, 4);    // 该操作会使得密钥 1 作废
        System.out.println(cache.get(1));       // 返回 -1 (未找到)
        System.out.println(cache.get(3));       // 返回  3
        System.out.println(cache.get(4));       // 返回  4
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
