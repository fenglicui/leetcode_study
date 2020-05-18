/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: solution
 * Author:   cfl
 * Date:     2019/8/2 10:09
 * Description: LeetCode 初级算法 链表相关题目
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com.linkedlist;

import java.util.ArrayList;
import java.util.List;

/**
 * 〈一句话功能简述〉<br>
 * 〈LeetCode 初级算法 链表相关题目〉
 *
 * @author cfl
 * @create 2019/8/2
 * @since 1.0.0
 */
public class solution {
    public static ListNode removeNthFromEnd(ListNode head, int n, int count, int[] len) {
        if (head == null)
            return null;
        len[0] += 1;
        head.next = removeNthFromEnd(head.next, n, count + 1, len);
        if (count == len[0] - n) {
            if (head.next == null)
                head = null;
            else {
                head.val = head.next.val;
                head.next = head.next.next;
            }
        }
        return head;
    }

    /**
     * Leetcode 从链表中删除倒数第n个节点
     *
     * @param head
     * @param n
     * @return
     */
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        int[] len = new int[1];
        return removeNthFromEnd(head, n, 0, len);
    }

    /**
     * LeetCode 反转链表
     *
     * @param head
     * @return
     */
    public static ListNode reverseList(ListNode head) {
        if (head == null) return null;
        ListNode prev = null;
        ListNode tmp = null;
        while (head != null) {
            tmp = prev;
            prev = new ListNode(head.val);
            prev.next = tmp;
            head = head.next;
        }
        return prev;
    }

    // 206. 反转链表 递归反转
//    ListNode header = null;
//    public ListNode reverseList(ListNode head, ListNode prev){
//        // 到达链表尾部
//        if (head == null && prev.next == null){
//            ListNode node = new ListNode(prev.val);
//            header = node;
//            return node;
//        }
//        ListNode tmp = reverseList(head.next, head);
//        tmp.next = new ListNode(prev.val);
//        tmp = tmp.next;
//        return tmp;
//    }
//
//    public ListNode reverseList(ListNode head) {
//        if (head == null || head.next == null) return head;
//        reverseList(head.next, head);
//        return header;
//    }

    //LeetCodeCode 909 两数相加
//    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
//        //添加头结点，head指向l 便于最后返回
//        ListNode l = new ListNode(-1);
//        ListNode head = l;
//        int c = 0;
//        // 首先计算两个链表相同长度的部分，注意进位
//        while (l1 != null && l2 != null) {
//            int n1 = l1.val, n2 = l2.val;
//            int sum = n1 + n2 + c;
//            ListNode tmp = new ListNode(sum % 10);
//            c = sum / 10;
//            l.next = tmp;
//            l1 = l1.next;
//            l2 = l2.next;
//            l = l.next;
//        }
//        // 然后对于l1或l2剩下的部分加到求和链表l上，注意产生进位
//        //把剩下的加上来
//        ListNode last = (l1 != null) ? l1 : l2;
//        while (last != null) {
//            int sum = last.val + c;
//            ListNode tmp = new ListNode(sum % 10);
//            c = sum / 10;
//            l.next = tmp;
//            last = last.next;
//            l = l.next;
//        }
//        // 最后如果首位有进位，添加一个节点
//        //如果最后c等于1 表示最高位进了1
//        if (c == 1) {
//            ListNode tmp = new ListNode(1);
//            l.next = tmp;
//        }
//        return head.next;
//    }

    public static ListNode removeZeroSumSublists(ListNode head) {
        List<Integer> values = new ArrayList<>();
        while (head != null) {
            values.add(head.val);
            head = head.next;
        }
        for (int i = 0; i < values.size() - 1; i++) {
            if (values.get(i) + values.get(i + 1) == 0) {
                values.remove(i);
                values.remove(i);
                i -= 2;
            }
        }
        ListNode root = new ListNode(-1);
        ListNode header = root;
        int i = 0;
        while (i < values.size()) {
            ListNode tmp = new ListNode(values.get(i++));
            root.next = tmp;
            root = root.next;
        }
        return header.next;
    }

    //Leetcode 环形链表 快慢指针法  https://blog.csdn.net/qq_33297776/article/details/81034628
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow)
                return true;
        }
        return false;
    }

    //Leetcode 回文链表 https://blog.csdn.net/sunao2002002/article/details/46918645
    public static boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        if (head != null && head.next == null) return true;
        ListNode slow = head;
        ListNode fast = head;
        //用快慢指针 快指针走完时，慢指针走完一半
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //判断奇偶
        if (fast != null) {  //长度为奇数
            slow = slow.next;
        }
        //反转slow
        slow = reverseList(slow);
        while (slow != null) {
            if (head.val != slow.val)
                return false;
            head = head.next;
            slow = slow.next;
        }
        return true;
    }

    //leetcode 排序链表 直接处理很麻烦时间复杂度较高版
//    public ListNode sortList(ListNode head) {
//        if (head == null || head.next == null)
//            return head;
//
//        ListNode root = head;
//        ListNode header = new ListNode(-1);
//        header.next = root;
//
//        List<Integer> nodes = new ArrayList<Integer>();
//        while (head != null){
//            nodes.add(head.val);
//            head = head.next;
//        }
//        Object[] values = nodes.toArray();
//        Arrays.sort(values);
//
//        for (Object value: values){
//            root.val = (int)value;
//            root = root.next;
//        }
//        return header.next;
//    }

    //leetcode 排序链表 对链表进行归并排序版
    public static ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        //设置快慢指针
        ListNode slow = head;
        ListNode fast = head.next;
        //遍历链表 当快指针走完链表时，慢指针恰好走了链表的一半
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //对链表后半部分进行排序
        ListNode right = sortList(slow.next);
        //截掉链表后半部分，只保留前半部分
        slow.next = null;
        //对前半部分进行排序
        ListNode left = sortList(head);
        return mergeList(left, right);
    }

    // leetcode 合并链表
    public static ListNode mergeList(ListNode l, ListNode r) {
        if (l == null && r == null) return null;
        ListNode res = new ListNode(-1);
        ListNode head = res;
        ;
        while (l != null || r != null) {
            ListNode tmp;
            if (l == null) {
                tmp = new ListNode(r.val);
                r = r.next;
            } else if (r == null) {
                tmp = new ListNode(l.val);
                l = l.next;
            } else if (l.val < r.val) {
                tmp = new ListNode(l.val);
                l = l.next;
            } else {
                tmp = new ListNode(r.val);
                r = r.next;
            }
            res.next = tmp;
            res = res.next;
        }
        return head.next;
    }

    // leetcode 相交链表
    // 先求出两个链表的长度，然后长的走到一样长，开始比较，能够找到相同的节点则返回，否则返回null
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
            return null;
        ListNode a = headA, b = headB;
        int m = 0, n = 0;
        while (a.next != null) {
            a = a.next;
            m++;
        }
        while (b.next != null) {
            b = b.next;
            n++;
        }
        if (a != b) return null;
        if (m > n) { // 交换二者长度，使得m<n
            ListNode tmpnode = headB;
            headB = headA;
            headA = tmpnode;
            int tmp = n;
            n = m;
            m = tmp;
        }
        while (m < n) {
            headB = headB.next;
            n--;
        }
        while (headA != null && headB != null) {
            if (headA == headB)
                return headA;
            headA = headA.next;
            headB = headB.next;
        }
        return null;
    }

    // leetcode 删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        ListNode head0 = new ListNode(-1);
        head0.next = head; // 添加头结点
        while (head != null && head.next != null) { // 如果至少还有两个节点
            int val = head.val;
            if (head.next.val == val) { // 出现了重复节点
                ListNode tmp = head.next;
                // 从next出发，向后删除重复节点
                while (tmp != null && tmp.val == val) {
                    tmp = tmp.next; // 删除重复节点
                }
                head.next = tmp;
            }
            head = head.next;
        }
        return head0.next;
    }


    // leetcode 删除列表中的元素
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        ListNode header = new ListNode(-1);
        // 删除开头的相同val
        while (head != null && head.val == val) {
            head = head.next;
        }
        header.next = head;
        // 最后一个为val的话前面一定有不为val的
        while (head != null) {
            // 一直往后删除等于val的node
            while (head.next != null && head.next.val == val)
                head.next = head.next.next;
            head = head.next;
        }
        return header.next;
    }

    // 876. 链表的中间结点
//    数组法
//    public ListNode middleNode(ListNode head) {
//        List<ListNode> nodes = new ArrayList<>();
//        // 遍历并保存节点，注意要创建一个临时节点，不然最后保留的都是同一个节点
//        while (head != null){
//            ListNode tmp = head;
//            nodes.add(tmp);
//            head = head.next;
//        }
//        // 链表长度为奇数 5/2, 链表长度为偶数 6/2，都是下标为3的node
//        return nodes.get(nodes.size()/2);
//    }

    public ListNode middleNode(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        // 快慢指针
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    //    445. 两数相加 II
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null)
            return null;
        if (l1 == null || l2 == null)
            return l1 == null ? l2 : l1;
        List<Integer> v1s = new ArrayList<>(), v2s = new ArrayList<>();
        while (l1 != null || l2 != null) {
            if (l1 != null) {
                v1s.add(0, l1.val);
                l1 = l1.next;
            }
            if (l2 != null) {
                v2s.add(0, l2.val);
                l2 = l2.next;
            }
        }
        ListNode node = null;
        int i = 0, j = 0, c = 0;
        while (i < v1s.size() || j < v2s.size() || c > 0) {
            int sum = 0;
            if (i < v1s.size()) {
                sum += v1s.get(i);
                i++;
            }
            if (j < v2s.size()) {
                sum += v2s.get(j);
                j++;
            }
            sum += c;
            ListNode newNode = new ListNode(sum % 10);
            c = sum / 10;
            newNode.next = node;
            node = newNode;
        }
        return node;
    }

    //    23. 合并K个排序链表 归并合并法
    public ListNode merge2Lists(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null)
            return l1 == null ? l2 : l1;
        ListNode header = new ListNode(-1);
        ListNode node = header;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                node.next = new ListNode(l1.val);
                l1 = l1.next;
            } else {
                node.next = new ListNode(l2.val);
                l2 = l2.next;
            }
            node = node.next;
        }
        if (l1 != null)
            node.next = l1;
        if (l2 != null)
            node.next = l2;
        return header.next;
    }

    public ListNode mergeKLists(ListNode[] lists, int left, int right) {
        if (left == right)
            return lists[left];
        int mid = left + (right - left) / 2;
        ListNode lNode = mergeKLists(lists, left, mid);
        ListNode rNode = mergeKLists(lists, mid + 1, right);
        return merge2Lists(lNode, rNode);
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0)
            return null;
        if (lists.length == 1)
            return lists[0];
        return mergeKLists(lists, 0, lists.length - 1);
    }

    public static void main(String[] args) {
//        ListNode head = new ListNode(1);
//        ListNode head2 = new ListNode(2);
//        ListNode head3 = new ListNode(3);
//        ListNode head4 = new ListNode(4);
//        ListNode head5 = new ListNode(5);
//        head.next = head2;
//        head2.next = head3;
//        head3.next = head4;
//        head4.next = head5;

//        head = removeNthFromEnd(head, 2);

//        head = removeNthFromEnd(head, 1);

//        head = reverseList(head);
//        while(head!=null){
//            System.out.println(head.val);
//            head = head.next;
//        }

//        ListNode head = new ListNode(1);
//        ListNode head2 = new ListNode(2);
//        ListNode head3 = new ListNode(-3);
//        ListNode head4 = new ListNode(3);
//        ListNode head5 = new ListNode(1);
//        head.next = head2;
//        head2.next = head3;
//        head3.next = head4;
//        head4.next = head5;
//
//        ListNode res = removeZeroSumSublists(head);
//        while(res!=null){
//            System.out.println(res.val);
//            res = res.next;
//        }

//        ListNode head = new ListNode(1);
//        ListNode head2 = new ListNode(2);
//        head.next = head2;
//        System.out.println(isPalindrome(head));

        ListNode head = new ListNode(-1);
        ListNode head2 = new ListNode(5);
        ListNode head3 = new ListNode(3);
        ListNode head4 = new ListNode(4);
        ListNode head5 = new ListNode(0);
        head.next = head2;
        head2.next = head3;
        head3.next = head4;
        head4.next = head5;

        head = sortList(head);
        while (head != null) {
            System.out.print(head.val + " ");
            head = head.next;
        }

    }
}
