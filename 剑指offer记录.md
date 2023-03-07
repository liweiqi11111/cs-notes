# leetcode 剑指offer

## **LinkedList**

- 面试题06-从尾到头打印链表

  > 辅助栈或递归

- 面试题22-链表中倒数第k个结点

  > 双指针维护一个间隔长度k，使用双指针就不用遍历获取链表长度

- 面试题24-反转链表

  >和题6的区别在于题6只要求返回节点的值；这题可以通过双指针来反转相邻节点的方向
  >
  >```java
  >// 输入: 1->2->3->4->5->NULL
  >// 输出: 5->4->3->2->1->NULL
  >public ListNode reverseList(ListNode head) {
  >    ListNode prev = null;
  >    ListNode cur = head;
  >    while(cur != null) {
  >        ListNode next = cur.next; // 先获取cur的next节点，后面cur的next需要反转
  >        cur.next = prev;
  >        prev = cur;
  >        cur = next;
  >    }
  >    return prev;
  >}
  >```

- 面试题25-合并两个排序的链表

- 面试题35-复杂链表的复制

  > 作好原链表和复制链表的节点之间的映射：HashMap

- 面试题52-两个链表的第一个公共节点

  > “浪漫相遇”

- 面试题18-删除链表的节点

  > 双指针：一个cur指针，一个prev指针



## **Stack & Queue**

- 面试题09-用两个栈实现队列

- 面试题30-包含min函数的栈

  > 维护一个**单调栈**

- 面试题31-栈的压入、弹出序列

  > 栈的模拟，以栈压入数组来进行遍历，这样写更简单、清晰

- 面试题58-1-翻转单词顺序

  > String去除首尾空格：trim()/strip()；分割：split()

- 面试题59-1-滑动窗口的最大值

  > 维护一个**单调队列**(由于两端都需要进行操作，使用deque)，deque内部仅包含当前滑动窗口中的元素下标，且使deque内这些下标对应的元素值单调递减
  >
  > 具体的维护操作：
  >
  > - 遍历给定数组中的元素，如果队列不为空且当前考察元素大于等于队尾元素，则将队尾元素移除。直到，队列为空或当前考察元素小于新的队尾元素；
  > - 将当前考察元素的下标加入队尾；
  > - 当队首元素的下标小于滑动窗口左侧边界left(right-k+1)时，表示队首元素已经不再滑动窗口内，因此将其从队首移除。
  > - 由于数组下标从0开始，因此当窗口右边界right+1大于等于窗口大小k时，意味着窗口形成。此时，队首元素就是该窗口内的最大值。

  ```java
  public int[] maxSlidingWindow(int[] nums, int k) {
  	int[] res = new int[nums.length - k + 1];
      LinkedList<Integer> queue = new LinkedList<>();
  
      for(int right = 0; right < nums.length; right++) {
          while(!queue.isEmpty() && nums[right] >= nums[queue.peekLast()]) {
              queue.removeLast();
          }
          queue.addLast(right);
          int left = right - k + 1;
          if(queue.peekFirst() < left) {
              queue.removeFirst();
          }
          if(right + 1 >= k) {
              res[left] = nums[queue.peekFirst()];
          }
      }
      return res;   
  }
  ```

  



## Tree

- [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

  > BFS——队列

- [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

  >1、BFS：与上一题不同，这一题使用双层循环，实现树的每一层所有节点同时入队、同时出队
  >
  >2、DFS：递归遍历树的同时，标记节点所处的层级数，将节点值放入对应的层级数组中

- [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode.cn/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

  >BFS——双端队列，在上一题的基础上，偶数层：插入队尾，奇数层：插入队首

- [剑指 Offer 26. 树的子结构](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

  >recur函数：以当前节点A作为根节点是否满足B为A的子结构；
  >
  >接下来采用BFS或DFS来遍历所有节点，调用recur函数进行判断
  >
  >```java
  >public boolean isSubStructure(TreeNode A, TreeNode B) {
  >        return (A != null && B != null) && (recur(A, B) || 
  >               isSubStructure(A.left, B) || isSubStructure(A.right, B));
  >    }
  >    boolean recur(TreeNode A, TreeNode B) {
  >        if(B == null) return true;
  >        if(A == null || A.val != B.val) return false;
  >        return recur(A.left, B.left) && recur(A.right, B.right);
  >    }
  >```

- [剑指 Offer 27. 二叉树的镜像](https://leetcode.cn/problems/er-cha-shu-de-jing-xiang-lcof/)

  >递归遍历每个节点，交换其左右子节点

- [剑指 Offer 28. 对称的二叉树](https://leetcode.cn/problems/dui-cheng-de-er-cha-shu-lcof/)

  >上一题是形成某个树的镜像，而该题是判断树是否是对称(原树与镜像树一致)，仍可以采取递归的方法

- [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode.cn/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

  > 回溯法：先序遍历+路径记录——对于当前节点，将其记录到路径中，如果值匹配将其记录到res数组，接下来递归读取其左右节点，最后将其从路径中删除
  >
  > ```java
  > LinkedList<List<Integer>> res = new LinkedList<>();
  > LinkedList<Integer> path = new LinkedList<>(); 
  > public List<List<Integer>> pathSum(TreeNode root, int sum) {
  >     recur(root, sum);
  >     return res;
  > }
  > void recur(TreeNode root, int tar) {
  >     if(root == null) return;
  >     path.add(root.val);
  >     tar -= root.val;
  >     if(tar == 0 && root.left == null && root.right == null)
  >         res.add(new LinkedList(path));
  >     recur(root.left, tar);
  >     recur(root.right, tar);
  >     path.removeLast();
  > }
  > ```

- [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

  >三个要素：
  >
  >1. 排序链表： 节点应从小到大排序，因此应使用 中序遍历 “从小到大”访问树的节点。
  >2. 双向链表： 在构建相邻节点的引用关系时，设前驱节点 `pre` 和当前节点 `cur` ，不仅应构建 `pre.right` = `cur` ，也应构建 `cur.left` = `pre` 。
  >3. 循环链表： 设链表头节点 `head` 和尾节点 `tail` ，则应构建 `head.left = tail` 和 `tail.right = head` 。
  >
  >```java
  >Node pre, head; // pre最后成为链表最后的节点，head为头节点
  >public Node treeToDoublyList(Node root) {
  >    if(root == null) return null;
  >    dfs(root);  // 构建要素1和2
  >    // 构建要素3
  >    head.left = pre;
  >    pre.right = head;
  >    return head;
  >}
  >// 中序遍历实现要素1，双指针：pre和cur用于构建指针连接关系从而实现要素2
  >void dfs(Node cur) {
  >    if(cur == null) return;
  >    dfs(cur.left);
  >    if(pre != null) pre.right = cur;
  >    else head = cur;
  >    cur.left = pre;
  >    pre = cur;
  >    dfs(cur.right);
  >}
  >```

- [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

  >1、优先队列
  >
  >2、利用二叉搜索树的性质：二叉搜索树的中序遍历为 **递增序列**，则二叉搜索树的中序遍历**倒序**为**递增序列**

- [剑指 Offer 55 - I. 二叉树的深度](https://leetcode.cn/problems/er-cha-shu-de-shen-du-lcof/)

- [剑指 Offer 55 - II. 平衡二叉树](https://leetcode.cn/problems/ping-heng-er-cha-shu-lcof/)

  > 1、从顶至底：先序遍历+判断深度
  >
  > 2、从底至顶：后序遍历+剪枝，本题最优解
  >
  > ```java
  > public boolean isBalanced(TreeNode root) {
  >     return recur(root) != -1;
  > }
  > private int recur(TreeNode root) {
  >     if (root == null) return 0;
  >     int left = recur(root.left);
  >     if(left == -1) return -1;  // 剪枝
  >     int right = recur(root.right);
  >     if(right == -1) return -1; // 剪枝
  >     // 这里在返回深度时进行了判断，方便进行剪枝操作
  >     return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1; 
  > }
  > ```

- [剑指 Offer 07. 重建二叉树](https://leetcode.cn/problems/zhong-jian-er-cha-shu-lcof/)

  > **分治算法**：递推参数的选择——根节点在前序遍历的索引 `root` 、子树在中序遍历的左边界 `left` 、子树在中序遍历的右边界 `right`

- [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

  > 分治，二叉搜索树的性质
  >
  > ```java
  > public boolean verifyPostorder(int[] postorder) {
  >     return recur(postorder, 0, postorder.length-1);
  > }
  > boolean recur(int[] postorder, int i, int j) {
  >     if(i >= j) return true;
  >     int p = i; 
  >     while(postorder[p] < postorder[j]) p++;
  >     int m = p; // 记录分割点，数组左子树均小于根节点，右子树均大于根节点
  >     while(postorder[p] > postorder[j]) p++;
  >     return p == j && recur(postorder, i, m-1) && recur(postorder, m, j-1);
  > }
  > ```
  >

- [剑指 Offer 37. 序列化二叉树](https://leetcode.cn/problems/xu-lie-hua-er-cha-shu-lcof/)

  > 使用层序遍历将二叉树进行序列化和反序列化