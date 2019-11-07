/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: Codec
 * Author:   cfl
 * Date:     2019/11/5 10:24
 * Description: leetcode：二叉树的序列化与反序列化
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com.Codec;


/**
 * 〈用层次遍历序列化和反序列化二叉树就是个！！傻逼！！！！！！〉<br>
 * 〈leetcode：二叉树的序列化与反序列化〉
 *
 * @author cfl
 * @create 2019/11/5
 * @since 1.0.0
 */
public class Codec {

    // 前序遍历
    public String preorder(TreeNode root) {
        if (root == null)
            return "n";
        return root.val + "#" + preorder(root.left) + "#" + preorder(root.right);
    }


    // Encodes a tree to a single string.
    // 按照前序遍历编码，节点之间以#隔开，null节点用n表示
    public String serialize(TreeNode root) {
        String s = "";
        if (root == null) return s;
        s = preorder(root);
        // System.out.println(s);
        return s;
    }

    int i = 0;

    public TreeNode build(String[] data) {
        if (i == data.length - 1 || data[i].equals("n")) return null;
        TreeNode root = new TreeNode(Integer.valueOf(data[i]));
        // System.out.println(serialize(root));
        i++;
        root.left = build(data);
        // System.out.println(serialize(root));
        i++;
        root.right = build(data);
        // System.out.println(serialize(root));
        return root;
    }

    // Decodes your encoded data to tree.
    // 前序遍历建立二叉树
    public TreeNode deserialize(String data) {
        if (data.length() == 0) return null;
        String[] nodes = data.split("#");
        TreeNode root = build(nodes);
        return root;
    }

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));

    public static void main(String[] args) {
        String nodes = "[5,2,3,null,null,2,4,3,1]";
        Codec codec = new Codec();
        System.out.println(codec.deserialize(nodes));
    }
}