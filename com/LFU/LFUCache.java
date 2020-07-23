package com.LFU;

import java.util.HashMap;

// LFU缓存，要求get和put的时间复杂度都是O(1)
// get操作是根据key获取键的value，时间复杂度为O(1),用哈希表
// put操作是插入或更新一个键值，注意容量有限。如果容量已满，需要删除使用频率最低且使用时间最远的键。
// 而每次get操作会改变键的访问频率，要想O(1)删除，需要保持一个以访问频率为key的哈希表。
// 既能够在O(1)时间内插入，又能在O(1)内删除，满足要求的数据结构就是双向链表。
// 所以以freq为key的哈希表，value保存对应访问频率的双向链表，每个节点要维护的信息包括key，value和freq。
// O(1)时间内get，维护一个以键为key的哈希表，value保存对应节点在freq哈希表保存的双向链表中的位置。
// 同时O(1)删除最小访问频率，维护一个minFreq的全局变量。


// 双链表节点类，包含key,value,freq,和前驱指针prev,后继指针next
class ListNode {
    int key;
    int value;
    int freq;
    ListNode prev;
    ListNode next;

    public ListNode(int key, int value, int freq) {
        this.key = key;
        this.value = value;
        this.freq = freq;
    }

}

// 双向链表类，包含头结点、尾结点，insert操作和delete操作
class DoubleLinkedList {
    ListNode head;
    ListNode tail;
    int freq;
    int size;

    public DoubleLinkedList(int freq) {
        this.freq = freq;
        head = new ListNode(-1, -1, 0);
        tail = new ListNode(-1, -1, 0);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    // 在链表头部插入节点
    public void insert(ListNode node) {
        node.freq += 1;
        head.next.prev = node;
        node.next = head.next;
        node.prev = head;
        head.next = node;
        size++;
    }

    // 删除指定节点
    public void delete(ListNode node) {
        node.next.prev = node.prev;
        node.prev.next = node.next;
        size--;
    }

    // 尾部删除
    public void delete() {
        delete(tail.prev);
        size--;
    }
}

class LFUCache {
    private int minFreq;
    private int capcity;
    private int size;

    private HashMap<Integer, ListNode> keyTable;
    private HashMap<Integer, DoubleLinkedList> freqTable;

    public LFUCache(int capacity) {
        this.capcity = capacity;
        minFreq = 0;
        size = 0;
        keyTable = new HashMap<>();
        freqTable = new HashMap<>();
    }

    public int get(int key) {
        if (!keyTable.containsKey(key))  // 不包含直接返回-1
            return -1;
        // 在原有freq的双向链表中删除node
        ListNode node = keyTable.get(key);
        DoubleLinkedList doubleLinkedList = freqTable.get(node.freq);
        doubleLinkedList.delete(node);
        freqTable.put(key, doubleLinkedList);
        // 在freq+1对应的双向链表中添加node
        doubleLinkedList = freqTable.getOrDefault(node.freq + 1, new DoubleLinkedList(node.freq + 1));
        doubleLinkedList.insert(new ListNode(key, node.value, node.freq + 1));
        freqTable.put(node.freq + 1, doubleLinkedList);
        // 更新最小访问频率
        if (freqTable.get(node.freq).size == 0 && minFreq == node.freq) {
            minFreq = node.freq + 1;
            freqTable.remove(node.freq);
        }
        ;
        return node.value;
    }

    public void put(int key, int value) {
        ListNode node;
        // 如果包含key，则更新keyTable中node的freq，在旧的freq对应双向链表中删除node
        if (keyTable.containsKey(key)) {
            node = keyTable.get(key);
            keyTable.put(key, new ListNode(key, value, node.freq + 1));
            DoubleLinkedList doubleLinkedList = freqTable.get(node.freq);
            doubleLinkedList.delete(node);
            freqTable.put(key, doubleLinkedList);
        } else {
            // 否则在keyTable中new一个node
            node = new ListNode(key, value, 0);
            keyTable.put(key, node);
            size++;
        }
        // 在新的freq中添加node
        DoubleLinkedList newFreqLinkList = freqTable.getOrDefault(node.freq + 1, new DoubleLinkedList(node.freq + 1));
        newFreqLinkList.insert(new ListNode(key, value, node.freq + 1));
        freqTable.put(node.freq + 1, newFreqLinkList);
        // 更新minFreq
        if (freqTable.get(node.freq).size == 0 && minFreq == node.freq) {
            minFreq = node.freq + 1;
        }
        ;
        // 如果超过容量，则删除minFreq对应的node
        if (size > capcity) {
            DoubleLinkedList doubleLinkedList = freqTable.get(minFreq);
            // 删除最少访问最远时间的node
            doubleLinkedList.delete();
            freqTable.put(minFreq, doubleLinkedList);
            if (freqTable.get(minFreq).size == 0) {
                minFreq = 0;
            }
        }
    }

    public static void main(String[] args) {

    }
}
