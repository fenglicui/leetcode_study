/**
 * Copyright (C), 2015-2019, XXX有限公司
 * FileName: MinStack
 * Author:   cfl
 * Date:     2019/9/14 18:19
 * Description: 最小栈
 * History:
 * <author>          <time>          <version>          <desc>
 * 作者姓名           修改时间           版本号              描述
 */
package com.minStack;

import java.util.Stack;

/**
 * 〈一句话功能简述〉<br>
 * 〈最小栈〉
 *
 * @author cfl
 * @create 2019/9/14
 * @since 1.0.0
 */
public class MinStack {
    private Stack<Integer> stack;  // 元素栈
    private Stack<Integer> minStack;  // 最小栈

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int x) {
        stack.push(x);
        // 如果最小栈为空，或者当前x比最小栈栈顶元素更小，则最小栈push
        if (minStack.empty() || minStack.peek() >= x)
            minStack.push(x);
    }

    public void pop() {
        // 如果要pop的是最小元素，那么最小栈也执行pop操作
        if (stack.peek().equals(minStack.peek()))
            minStack.pop();
        stack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
