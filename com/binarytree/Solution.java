/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: Solution
 * Author:   cfl
 * Date:     2019/8/30 16:49
 * Description: 二叉树相关
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com.binarytree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

/**
 * 〈一句话功能简述〉<br>
 * 〈二叉树相关〉
 *
 * @author cfl
 * @create 2019/8/30
 * @since 1.0.0
 */
public class Solution {
    public boolean isValidBST(TreeNode root, long minValue, long maxValue) {
        if (root == null) return true;
        if (root.val <= minValue || root.val >= maxValue)
            return false;
        boolean left = isValidBST(root.left, minValue, root.val);
        boolean right = isValidBST(root.right, root.val, maxValue);
        return left && right;
    }

    //Leetcode 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    //LeetCode 对称二叉树 迭代版本
//    public static boolean isSymmetric(TreeNode root) {
//        if (root == null) return false;
//        List<TreeNode> nodes = new ArrayList<TreeNode>();
//        nodes.add(root);
//        while (nodes.size()>0) {
//            List<TreeNode> tmps = new ArrayList<TreeNode>();
//            int count = 0;  //用于记录下一层节点数量
//            //扫描一遍当前节点集合，获取下一层节点集合
//            for (int i = 0; i < nodes.size(); i++) {
//                TreeNode tmp = nodes.get(i);
//                TreeNode empty = new TreeNode(Integer.MAX_VALUE);
//                if (tmp.left!=null){
//                    tmps.add(tmp.left);count++;
//                }
//                else tmps.add(empty);
//                if (tmp.right!=null){
//                    tmps.add(tmp.right);count++;
//                }
//                else tmps.add(empty);
//            }
//            //如果左右节点之和为奇数
//            if (count % 2 > 0){
//                return false;
//            }
//            //所有节点遍历结束
//            if (count == 0)
//                break;
//            int tmps_size = tmps.size();
//            for (int i = 0; i < tmps_size/2; i++){
//                if (tmps.get(i).val != tmps.get(tmps_size-1-i).val)
//                    return false;
//            }
//            nodes = new ArrayList<TreeNode>(tmps);
//        }
//        return true;
//    }

    //LeetCode 对称二叉树 递归版本
    public static boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetric(root.left, root.right);
    }

    public static boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isSymmetric(left.right, right.left) && isSymmetric(left.left, right.right);
    }

    // leetcode 二叉树的层次遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> nodes = new ArrayList<>();
        if (root == null) return nodes;
        //创建第一层
        List<TreeNode> layers = new ArrayList<>();
        layers.add(root);
        List<Integer> values = new ArrayList<>();
        values.add(root.val);
        //添加根节点
        nodes.add(values);
        while (layers.size() > 0) {
            List<TreeNode> tmps = new ArrayList<>();
            values = new ArrayList<>();
            for (int i = 0; i < layers.size(); i++) {
                TreeNode node = layers.get(i);
                if (node.left != null) {
                    tmps.add(node.left);
                    values.add(node.left.val);
                }
                if (node.right != null) {
                    tmps.add(node.right);
                    values.add(node.right.val);
                }
            }
            if (tmps.size() == 0) break;
            nodes.add(values);
            layers = tmps;
        }
        return nodes;
    }

    public TreeNode sortedArrayToBST(int[] nums, int left, int right) {
        if (left == right)
            return new TreeNode(nums[left]);
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        if (mid > left) root.left = sortedArrayToBST(nums, left, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, right);
        return root;
    }

    // leetcode 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) return null;
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    // leetcode 中序遍历二叉树 迭代写法
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> nodes = new ArrayList<>();
        if (root == null) return nodes;
        Stack<TreeNode> stack = new Stack<>();
        // 1表示为当前访问的左子树节点，2表示为当前访问的右子树节点，0表示当前访问的根节点
        Stack<Integer> lrs = new Stack<>();
        // 初始化两个栈，先将root加到栈中
        TreeNode node = root;
        stack.push(node);
        lrs.push(0);
        // 第一次从root开始遍历的标记，因为第一次必须把左右子树检查完
        boolean start = true;
        // 栈为空时，所有节点都遍历完
        while (!stack.empty()) {
            node = stack.peek();
            int flag = lrs.peek();
            // 先将当前节点从栈中移出
            stack.pop();
            lrs.pop();
            //lrs.peek()=0 当前节点的左右子树都访问过,添加到nodes中
            // 当前节点为最后一层的节点，直接添加到nodes
            if ((flag == 0 && !start) || (node.left == null && node.right == null)) {
                nodes.add(node.val);
                continue;
            }
            if (node.right != null) {
                stack.push(node.right);
                lrs.push(2);
            }
            stack.push(node);
            lrs.push(0);
            if (node.left != null) {
                stack.push(node.left);
                lrs.push(1);
            }
            start = false;
        }
        return nodes;
    }

    //     leetcode 填充每个节点的下一个右侧节点指针
    // Definition for a Node.
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    // leetcode 填充每个节点的下一个右侧节点指针
    public Node connect(Node root) {
        if (root == null || root.left == null)
            return root;
        // 一次性处理left right 两个节点，每次传到下一层时，当前节点都已处理过next
        // root的next为null，也即不需要处理
        root.left.next = root.right;
        // 如果root有next
        if (root.next != null)
            root.right.next = root.next.left;
        connect(root.left);
        connect(root.right);
        return root;
    }

    public TreeNode buildBinaryTree(List<Integer> preorder, List<Integer> inorder) {
//        记录左子树节点个数
//        不用ArrayList，直接用数组，标记起始位置会更快
        if (preorder.size() == 0)
            return null;
        int v = preorder.get(0);
        TreeNode root = new TreeNode(v);
        if (preorder.size() == 1)
            return root;
        int idx = inorder.indexOf(v);
        int left_num = 0;
        if (idx > 0) {
            left_num = idx;
            root.left = buildBinaryTree(preorder.subList(1, left_num + 1), inorder.subList(0, idx));
        }
        if (idx < inorder.size() - 1) {
            root.right = buildBinaryTree(preorder.subList(left_num + 1, preorder.size()), inorder.subList(idx + 1, inorder.size()));
        }
        return root;
    }

    // leetcode 从前序与中序遍历序列构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // int[] to List<Integer>
        List<Integer> pre = Arrays.stream(preorder).boxed().collect(Collectors.toList());
        List<Integer> in = Arrays.stream(inorder).boxed().collect(Collectors.toList());
        return buildBinaryTree(pre, in);
    }

    // leetcode 二叉搜索树中第k小的元素
    // 中序遍历，是按照从小到大的顺序
    public static int value;
    public static int i;

    public static void inorder(TreeNode root, int k) {
        if (root == null) return;
        inorder(root.left, k);
        i++;
        if (i == k) {
            value = root.val;
            return;
        }
        inorder(root.right, k);
    }

    public static int kthSmallest(TreeNode root, int k) {
        i = value = 0;
        inorder(root, k);
        return value;
    }

    // leetcode 二叉树的锯齿形层次遍历
    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> results = new ArrayList<>();
        if (root == null) return results;
        List<Integer> result = new ArrayList<>();
        result.add(root.val);
        results.add(result);
        List<TreeNode> layer = new ArrayList<>();
        layer.add(root);
        List<TreeNode> tmp = new ArrayList<>();

        boolean left2right = false;

        while (true) {
            result = new ArrayList<>();
            tmp = new ArrayList<>();
            //从左到右遍历当前层子节点
            // 注意 上一层节点以从右到左方式存入，这里节点集合要倒序遍历
            if (left2right) {
                for (int i = layer.size() - 1; i >= 0; i--) {
                    if (layer.get(i).left != null) {
                        result.add(layer.get(i).left.val);
                        tmp.add(layer.get(i).left);
                    }
                    if (layer.get(i).right != null) {
                        result.add(layer.get(i).right.val);
                        tmp.add(layer.get(i).right);
                    }
                }
            } else {
                //从右到左遍历当前层子节点
                // 注意 上一层节点以从左到右方式存入，这里节点集合要倒序遍历
                for (int i = layer.size() - 1; i >= 0; i--) {
                    if (layer.get(i).right != null) {
                        result.add(layer.get(i).right.val);
                        tmp.add(layer.get(i).right);
                    }
                    if (layer.get(i).left != null) {
                        result.add(layer.get(i).left.val);
                        tmp.add(layer.get(i).left);
                    }
                }
            }
            if (tmp.isEmpty()) break;
            left2right = !left2right;
            layer = tmp;
            results.add(result);
        }
        return results;
    }

    public static void main(String[] args) {
//        TreeNode root = new TreeNode(1);
//        TreeNode node1 = new TreeNode(2);
//        TreeNode node2 = new TreeNode(3);
//        node1.left = node2;
//        root.right = node1;
//        isSymmetric(root);
//        System.out.println(inorderTraversal(root));

//        TreeNode root = new TreeNode(3);
//        TreeNode node1 = new TreeNode(1);
//        TreeNode node2 = new TreeNode(4);
//        TreeNode node3 = new TreeNode(2);
//        node1.right = node3;
//        root.left = node1;
//        root.right = node2;
//        System.out.println(kthSmallest(root, 1));

        TreeNode root = new TreeNode(1);
        TreeNode node1 = new TreeNode(2);
        TreeNode node2 = new TreeNode(3);
        TreeNode node3 = new TreeNode(4);
        TreeNode node4 = new TreeNode(5);
        node1.left = node3;
        node2.right = node4;
        root.left = node1;
        root.right = node2;
        System.out.println(zigzagLevelOrder(root));

    }
}
