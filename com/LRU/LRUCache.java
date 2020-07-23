package com.LRU;

import java.util.HashMap;


// leetcode 哈希表+双向链表 哈希链表 实现LRU缓存机制
class Node {
    int key;
    int val;
    Node prev;
    Node next;

    public Node(int k, int v) {
        key = k;
        val = v;
    }
}

class DoubleLinkedList {
    Node head;
    Node tail;
    int capacity;

    public DoubleLinkedList() {
        this.capacity = 0;
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head.next = tail;
        tail.prev = head;
    }

    public void insertToTail(Node node) {
        node.prev = tail.prev;
        node.next = tail;
        tail.prev.next = node;
        tail.prev = node;
        capacity++;
    }

    public void deleteNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
        capacity--;
    }

    public Node deleteFirst() {
        Node res = head.next;
        deleteNode(head.next);
        return res;
    }
}
class LRUCache {
    int capacity;
    DoubleLinkedList doubleLinkedList;
    HashMap<Integer, Node> map;
    public LRUCache(int capacity) {
        this.capacity = capacity;
        doubleLinkedList = new DoubleLinkedList();
        map = new HashMap<>();
    }

    // 如果key存在，则更新链表，返回值，否则返回-1
    public int get(int key) {
        if (!map.containsKey(key))
            return -1;
        Node node = map.get(key);
        doubleLinkedList.deleteNode(node);
        doubleLinkedList.insertToTail(node);
        return node.val;
    }

    // 如果key存在，则更新node的value，删除原来的node，插入到链表尾部
    // 否则插入新建的node到链表尾部，然后判断容量，如果大于上限则删除链表头部第一个node
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            doubleLinkedList.deleteNode(node);
            node.val = value;
            doubleLinkedList.insertToTail(node);// 这里要用原来的node，因为map存储的是node的地址

        } else {
            Node node = new Node(key, value);
            doubleLinkedList.insertToTail(node);
            map.put(key, node);  // 这里要插入之后再put，这样node的prev和next都有效
            int size = doubleLinkedList.capacity;
            if (size > capacity) {
                node = doubleLinkedList.deleteFirst();
                // node设计包含key，作用就在这 保证删除第一个node能O(1)时间内更新map
                map.remove(node.key);
            }
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
