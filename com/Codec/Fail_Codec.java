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


import java.util.ArrayList;
import java.util.List;

/**
 * 〈用层次遍历序列化和反序列化二叉树就是个！！傻逼！！！！！！〉<br>
 * 〈leetcode：二叉树的序列化与反序列化〉
 *
 * @author cfl
 * @create 2019/11/5
 * @since 1.0.0
 */
public class Fail_Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {

        StringBuffer res = new StringBuffer("[");

        if (root == null) {
            res.append("]");
            return res.toString();
        }

        res.append(root.val);

        List<TreeNode> layer = new ArrayList<TreeNode>();
        List<TreeNode> tmp = new ArrayList<TreeNode>();
        tmp.add(root);

        while (!tmp.isEmpty()) {
            layer = tmp;
            tmp = new ArrayList<TreeNode>();
            for (TreeNode node : layer) {
                if (node.left != null) {
                    tmp.add(node.left);
                    res.append(",");
                    res.append(node.left.val);
                } else res.append(",null");
                if (node.right != null) {
                    tmp.add(node.right);
                    res.append(",");
                    res.append(node.right.val);
                } else res.append(",null");
            }
        }

        // System.out.println(layer.size());
        //删掉多余的null 最后的layer为二叉树最后一层的节点，都多生成了2个null节点，每个节点长度为5 即,null
        res.delete(res.length() - layer.size() * 5 * 2, res.length());
        res.append("]");
//        System.out.println(res.toString());
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public static TreeNode deserialize(String data) {
        // 去掉首尾的[]
        data = data.substring(1, data.length() - 1);
//        System.out.println(data);
        if (data.length() == 0)
            return null;
        String[] nodes = data.split(",");
        TreeNode root = null;
        // i nodes下标指针
        int i = 0;
        // 初始化根节点
        root = new TreeNode(Integer.valueOf(nodes[i]));
        i++;
        // 树的层数 要向上取整
        int n = (int) Math.ceil(Math.log(nodes.length + 1) / Math.log(2.0));

        List<TreeNode> layer = new ArrayList<>();
        layer.add(root);
        n--;

        List<TreeNode> tmplayer;
        while (n-- > 0) {
            tmplayer = new ArrayList<>();
            // 建立某一层子树
            for (TreeNode node : layer) {
                // 当前节点为空，添加两个null节点
                if (node == null) {
//                    i+=2;
                    tmplayer.add(null);
                    tmplayer.add(null);
                    continue;
                }

                String lval = nodes[i], rval = nodes[i + 1];

                TreeNode ln = null, rn = null;

                node.left = node.right = null;

                // 添加左子树
                if (!lval.equals("null")) {
                    node.left = ln = new TreeNode(Integer.valueOf(lval));
                }

                // 添加右子树
                if (!rval.equals("null")) {
                    node.right = rn = new TreeNode(Integer.valueOf(rval));
                }
                tmplayer.add(ln);
                tmplayer.add(rn);
                i += 2;
            }
            layer = tmplayer;
        }
        return root;
    }

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));

    public static void main(String[] args) {
        String nodes = "[5,2,3,null,null,2,4,3,1]";
        System.out.println(deserialize(nodes));
    }
}