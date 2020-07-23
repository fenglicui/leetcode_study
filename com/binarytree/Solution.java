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

import java.util.*;

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
    // 101. 对称二叉树 迭代
//    public boolean isSymmetric(TreeNode root) {
//        if (root == null)
//            return true;
//        List<Integer> layers = new ArrayList<>();
//        Deque<TreeNode> stack = new ArrayDeque<>();
//        stack.addLast(root);
//        while(!stack.isEmpty()){
//            int size = stack.size();
//            while(size-->0){
//                TreeNode node = stack.removeFirst();
//                if (node.left!=null){
//                    // System.out.println(node.left.val);
//                    stack.addLast(node.left);
//                    layers.add(node.left.val);
//                }else layers.add(Integer.MAX_VALUE);
//                if (node.right!=null){
//                    // System.out.println(node.right.val);
//                    stack.addLast(node.right);
//                    layers.add(node.right.val);
//                }else layers.add(Integer.MAX_VALUE);
//            }
//            int i = 0, j = layers.size()-1;
//            while(i<j){
//                // System.out.println(layers.get(i) + "," + layers.get(j));
//                if (!layers.get(i).equals(layers.get(j))){
//                    return false;
//                }
//                i++;j--;
//            }
//            layers = new ArrayList<>();
//
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

    //    105. 从前序与中序遍历序列构造二叉树
    HashMap<Integer, Integer> inorderMap;

    public TreeNode helper(int[] preorder, int pre_left, int pre_right, int[] inorder, int in_left, int in_right) {
        if (pre_left > pre_right)
            return null;
        TreeNode root = new TreeNode(preorder[pre_left]);
        int index = inorderMap.get(root.val);
        int left_num = index - in_left;
        root.left = helper(preorder, pre_left + 1, left_num + pre_left, inorder, in_left, index - 1);
        root.right = helper(preorder, left_num + pre_left + 1, pre_right, inorder, index + 1, in_right);
        return root;
    }
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0 || inorder.length == 0)
            return null;
        inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++)
            inorderMap.put(inorder[i], i);
        return helper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
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

    // LeetCode 二叉树的最近公共祖先
    private TreeNode ans;

    public Solution() {
        this.ans = null;
    }

    public boolean rescurTree(TreeNode root, TreeNode p, TreeNode q) {
        // 空节点返回false
        if (root == null) return false;
        // 递归左子树，返回是否存在p,q节点
        int left = rescurTree(root.left, p, q) ? 1 : 0;
        // 递归右子树，返回是否存在p,q节点
        int right = rescurTree(root.right, p, q) ? 1 : 0;
        // 当前节点是否为p q
        int mid = (root.val == p.val || root.val == q.val) ? 1 : 0;
        // System.out.println("left=" + left + ",right=" + right + ",mid=" + mid + ",root.val=" + root.val);
        // 如果在当前节点存在p和q，设置ans
        if (this.ans == null && left + mid + right >= 2)
            this.ans = root;
        // 返回是否存在p q节点
        return left + mid + right > 0;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        rescurTree(root, p, q);
        return this.ans;
    }

    // LeetCode 543.二叉树的直径
    int diameter = 0;

    public int diameterOfBiTree(TreeNode root) {
        if (root == null)
            return -1;
        // 计算左右子树到当前节点的路径
        int d1 = diameterOfBiTree(root.left) + 1;
        int d2 = diameterOfBiTree(root.right) + 1;
        // 如果穿过当前节点路径更长，更新diameter
        if (d1 + d2 > diameter) diameter = d1 + d2;
        return Math.max(d1, d2);
    }

    public int diameterOfBinaryTree(TreeNode root) {
        // 取最大值
        return Math.max(diameterOfBiTree(root), diameter);
    }

    // 竞赛题：将二叉搜索树变平衡
    boolean balance = true;

    // 中序遍历的同时判断是否平衡
    public int inOrder(List<Integer> vals, TreeNode root) {
        if (root == null)
            return -1;
        int left = inOrder(vals, root.left) + 1;
        vals.add(root.val);
        int right = inOrder(vals, root.right) + 1;
        if (left - right > 1 || right - left > 1)
            balance = false;
        return Math.max(left, right);

    }

    // 建立二叉搜索树
    public TreeNode creatBST(List<Integer> vals, int left, int right) {
        if (left > right)
            return null;
        // if (left == right)
        //     return new TreeNode(root.val);
        int mid = left + (right - left) / 2;
        TreeNode node = new TreeNode(vals.get(mid));
        node.left = creatBST(vals, left, mid - 1);
        node.right = creatBST(vals, mid + 1, right);
        return node;
    }

    public TreeNode balanceBST(TreeNode root) {
        if (root == null || (root.left == null && root.right == null))
            return root;
        List<Integer> vals = new ArrayList<>();
        inOrder(vals, root);
        if (balance)
            return root;
        // 不平衡的话，中序遍历得到的是升序数组，直接在此基础上建立二叉搜索树
        root = creatBST(vals, 0, vals.size() - 1);
        return root;
    }

    List<Integer> rightSideViewRes = new ArrayList<>();

    public void rightSideViewTrackBack(TreeNode root, int depth) {
        if (root == null)
            return;
        if (rightSideViewRes.size() < depth)
            rightSideViewRes.add(root.val);
        rightSideViewTrackBack(root.right, depth + 1);
        rightSideViewTrackBack(root.left, depth + 1);
    }

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null)
            return new ArrayList<Integer>();
        rightSideViewRes = new ArrayList<>();
        rightSideViewTrackBack(root, 1);
        return rightSideViewRes;
    }

    // 572. 另一个树的子树 判断t是否是s的子树：三种情况，s和t相等，t是s的左子树的子树，t是s的右子树的子树。
    // 判断相等，当然是val相等，左子树相等，且右子树相等。
    public boolean isSametree(TreeNode s, TreeNode t) {
        if (s == null && t == null)
            return true;
        if (s == null || t == null)
            return false;
        if (s.val != t.val)
            return false;
        return isSametree(s.left, t.left) && isSametree(s.right, t.right);
    }

    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null && t == null)
            return true;
        if (s == null && t != null)
            return false;
        return isSametree(s, t) || isSubtree(s.left, t) || isSubtree(s.right, t);
    }

    // 102. 二叉树的层序遍历 用队列实现
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root == null)
            return res;
        q.offer(root);
        while (q.size() > 0) {
            int n = q.size();
            List<Integer> vals = new ArrayList<>();
            while (n-- > 0) {
                TreeNode node = q.poll();
                vals.add(node.val);
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null)
                    q.offer(node.right);
            }
            res.add(vals);
        }
        return res;
    }

    // 95. 不同的二叉搜索树 II 递归搜索所有可能的左子树 和 右子树
    // 保证二叉搜索树的关键是左子树的值都小于当前节点，右子树的值大于当前节点
    // 以i为根节点，只需要1-i-1创建左子树，i+1-n创建右子树，这样便形成了递归
    public List<TreeNode> generateTrees(int n) {
        if (n <= 0)
            return new ArrayList<TreeNode>();
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> subLefts = generateTrees(start, i - 1);
            List<TreeNode> subRights = generateTrees(i + 1, end);
            for (TreeNode left : subLefts) {
                for (TreeNode right : subRights) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }
        return res;
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
