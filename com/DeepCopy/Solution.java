package com.DeepCopy;


import java.util.HashMap;
import java.util.Map;

// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}


class Solution {
    public Node copyRandomList(Node head) {
        if (head == null)
            return null;
        // 新旧节点的映射
        Map<Node, Node> nodeMap = new HashMap<>();
        Node old = head;
        while (head != null) {
            // 如果旧的节点访问过即存在于Map，则将当前节点指针 指向其对应的新节点
            if (!nodeMap.containsKey(head))
                nodeMap.put(head, new Node(head.val));
            if (head.random != null) {
                if (!nodeMap.containsKey(head.random))
                    nodeMap.put(head.random, new Node(head.random.val));
                nodeMap.get(head).random = nodeMap.get(head.random);
            }
            if (head.next != null) {
                if (!nodeMap.containsKey(head.next))
                    nodeMap.put(head.next, new Node(head.next.val));
                nodeMap.get(head).next = nodeMap.get(head.next);
            }
            head = head.next;
        }
        return nodeMap.get(old);
    }
}
