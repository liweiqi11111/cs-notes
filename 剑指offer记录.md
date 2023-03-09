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



## Heap

- [剑指 Offer 40. 最小的k个数](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/)

  > 1. 优先队列
  > 2. 快速排序
  > 3. 快速选择
  >
  > ```java
  > public int[] getLeastNumbers(int[] arr, int k) {
  >     if(k >= arr.length) return arr;
  >     return quickSelect(arr, k, 0, arr.length-1);
  > }
  > public int[] quickSelect(int[] arr, int k, int l, int r) {
  >     int i = l, j = r;
  >     while(i < j) {
  >         while(i < j && arr[j] >= arr[l]) j--;
  >         while(i < j && arr[i] <= arr[l]) i++;
  >         swap(arr, i, j);
  >     }
  >     swap(arr, l, i);
  >     if(i > k) return quickSelect(arr, k, l, i-1);
  >     if(i < k) return quickSelect(arr, k, i+1, r);
  >     return Arrays.copyOf(arr, k);
  > }
  > private void swap(int[] arr, int i, int j) {
  >     int tmp = arr[i];
  >     arr[i] = arr[j];
  >     arr[j] = tmp;
  > }
  > ```

## Map

- [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode.cn/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

  > 1. HashMap
  > 2. LinkedHashMap：有序哈希表中的键值对是 **按照插入顺序排序** 的。基于此，可通过遍历有序哈希表，实现搜索首个 “数量为 1 的字符”。
  >
  > ```java
  > public char firstUniqChar(String s) {
  >     Map<Character, Boolean> map = new LinkedHashMap<>();
  >     for(char l : s.toCharArray()){
  >         map.put(l, !map.containsKey(l));
  >     }
  >     for(Map.Entry<Character, Boolean> entry : map.entrySet()){
  >         if(entry.getValue()) return entry.getKey();
  >     }        
  >     return ' ';
  > }
  > ```



## Graph

- [剑指 Offer 12. 矩阵中的路径](https://leetcode.cn/problems/ju-zhen-zhong-de-lu-jing-lcof/)

  > 矩阵搜索问题：深度优先搜索 + 剪枝
  >
  > 注意：
  >
  > 1. 矩阵中的每个节点都可能作为起始点，所以主函数要对矩阵每个节点分别进行判断，只要有一个节点返回true就表示存在。
  > 2. 在dfs的过程中，可以使用visited数组来判断节点是否已遍历过，但在上面的条件下空间开销会增大，可以在原矩阵数组上进行操作：遍历过的节点修改值为`\0`，在递归调用结束后，恢复原来的值。
  >
  > ```java
  > public boolean exist(char[][] board, String word) {
  >     char[] words = word.toCharArray();
  >     for(int i = 0; i < board.length; i++) {
  >         for(int j = 0; j < board[0].length; j++) {
  >             if(dfs(board, words, i, j, 0)) return true;
  >         }
  >     }
  >     return false;
  > }
  > boolean dfs(char[][] board, char[] word, int i, int j, int k) {
  >     if(i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != word[k]) return false;
  >     if(k == word.length - 1) return true;
  >     board[i][j] = '\0';
  >     boolean res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
  >     board[i][j] = word[k];
  >     return res;
  > }
  > ```

- [面试题13. 机器人的运动范围](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

  > 和上一题类似，计算数位和时可以通过调用函数逐一计算，也可以通过下面的增量方法计算：
  >
  > 数位和增量公式：**`(x + 1) % 10 != 0 ? s_x + 1 : s_x - 8;`**
  >
  > ```java
  > int m, n, k;
  > boolean[][] visited;
  > public int movingCount(int m, int n, int k) {
  >     this.m = m; this.n = n; this.k = k;
  >     this.visited = new boolean[m][n];
  >     return dfs(0, 0, 0, 0);
  > }
  > public int dfs(int i, int j, int si, int sj) {
  >     if(i >= m || j >= n || k < si + sj || visited[i][j]) return 0;
  >     visited[i][j] = true;
  >     return 1 + dfs(i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj) + dfs(i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8);
  >     }
  > ```



## 具体算法类题目



### 搜索算法

- [剑指 Offer 04. 二维数组中的查找](https://leetcode.cn/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

  > 利用该二维数组行和列非递减的性质，可以选择从数组左上角或右下角来进行二分搜索
  >
  > ```java
  > public boolean findNumberIn2DArray(int[][] matrix, int target) {
  >     if(matrix.length == 0) return false;
  >     return searchNum(matrix, target, 0, matrix[0].length-1);
  > }
  > boolean searchNum(int[][] matrix, int target, int i, int j) {
  >     if(i >= matrix.length || j < 0) return false;
  >     if(matrix[i][j] > target) return searchNum(matrix, target, i, j-1);
  >     else if(matrix[i][j] < target) return searchNum(matrix, target, i+1, j);
  >     return true;
  > }
  > ```

- [剑指 Offer 11. 旋转数组的最小数字](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

  > 寻找旋转数组的最小元素即为寻找**右排序数组**的首个元素`numbers[x]`，称x为**旋转点**
  >
  > 排序数组的查找问题首先考虑使用**二分法**解决——$O(log_2 n)$
  >
  > ```java
  > public int minArray(int[] numbers) {
  >  int i = 0, j = numbers.length - 1;
  >  while (i < j) {
  >      int m = (i + j) / 2;
  >      if (numbers[m] > numbers[j]) i = m + 1;
  >      else if (numbers[m] < numbers[j]) j = m;
  >      else j--;
  >  }
  >  return numbers[i];
  > }
  > ```
  >
  > 注意：
  >
  > - 当`numbers[m] = numbers[j]`时，无法判断m在哪个排序数组中，可以执行`j--`缩小判断范围([证明](https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/solution/mian-shi-ti-11-xuan-zhuan-shu-zu-de-zui-xiao-shu-3/))。
  > - 为什么不用`numbers[m]`和`numbers[i]`作比较：在`numbers[m] > numbers[i]`情况下，无法判断m在哪个排序数组中。

-  [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

  > 要求时间复杂度 $O(N)$ ，空间复杂度$O(1)$ ，因此首先排除 **暴力法** 和 **哈希表统计法** 
  >
  > 1. nums数组整体异或计算出$x \bigotimes y$
  > 2. 获取$x \bigotimes y$结果的首位1的位置，记录于m中
  > 3. 根据第2步中的这一位对nums数组进行拆分
  > 4. 两个子数组分别异或求出x和y
  >
  > ```java
  > public int[] singleNumbers(int[] nums) {
  >     int x = 0, y = 0, n = 0, m = 1;
  >     for(int num : nums) {
  >         n ^= num;
  >     }
  >     while((n & m) == 0) {
  >         m <<= 1;
  >     }
  >     for(int num : nums) {
  >         if((num & m) != 0) x ^= num;
  >         else y ^= num;
  >     }
  >     return new int[] {x, y};
  > }
  > ```

- [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode.cn/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

  > 题目没有要求时间复杂度，但我们仍限制时间复杂度 $O(N)$ ，空间复杂度$O(1)$ ；
  >
  > 对于出现三次的数字，各**二进制位**出现的次数都是3的倍数，因此统计所有数字的各二进制位中1的出现次数，并对3求余，结果则为只出现一次的数字。
  >
  > 理解：每个二进制位，有限状态自动机(卡诺图法化简)
  >
  > ```java
  > public int singleNumber(int[] nums) {
  >     int ones = 0, twos = 0;
  >     for(int num : nums) {
  >         ones = ones ^ num & ~twos;
  >         twos = twos ^ num & ~ones;
  >     }
  >     return ones;
  > }
  > ```



### 动态规划

- [剑指 Offer 10- I. 斐波那契数列](https://leetcode.cn/problems/fei-bo-na-qi-shu-lie-lcof/)

  > 1. 递归法
  >
  > 2. 记忆化递归法
  >
  > 3. 动态规划——时间和空间最佳解法
  >
  >    `dp[i+1] = dp[i] + dp[i-1]`，`dp[0] = 0`, `dp[1] = 1`,返回`dp[n]`
  >
  >    - 空间复杂度优化：只需初始化三个整型变量`sum`，`a`，`b`，利用辅助变量`sum`使`a`和`b`交替前进
  >    - 循环求余法：随着n增大，f(n)会超过`Int32`甚至`Int64`，导致最终结果错误；可以利用求余运算法则，在循环的过程中就进行求余操作。
  >
  > ```java
  > public int fib(int n) {
  >     int a = 0, b = 1, sum;
  >     for(int i = 0; i < n; i++) {
  >        	sum = (a + b) % 1000000007;
  >         a = b;
  >         b = sum;
  >     }
  >     return a;
  > }
  > ```

- [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

- [剑指 Offer 63. 股票的最大利润](https://leetcode.cn/problems/gu-piao-de-zui-da-li-run-lcof/)

  > 状态定义：dp[i]代表以prices[i]为结尾的子数组的最大利润（简称为前i日的最大利润）
  >
  > 转移方程（初始：`dp[0] = 0`）：
  >
  > 前i日的最大利润 = max(前(i-1)日的最大利润，第i日价格-前i日最低价格)
  >
  > `dp[i] = max(dp[i-1], prices[i] - min(prices[0:i]))`
  >
  > 时间复杂度降低：在遍历的过程中更新前i日最低价格；
  >
  > 空间复杂度降低：使用一个变量`profit`代替dp列表
  >
  > ```java
  > public int maxProfit(int[] prices) {
  >     int cost = Integer.MAX_VALUE, profit = 0;
  >     for(int price : prices) {
  >         cost = Math.min(cost, price);
  >         profit = Math.max(profit, price - cost);
  >     }
  >     return profit;
  > }
  > ```

- [剑指 Offer 42. 连续子数组的最大和](https://leetcode.cn/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

  > 要求时间复杂度为O(n)——动态规划
  >
  > 状态定义：dp[i]代表以元素nums[i]为结尾的连续子数组最大和
  >
  > 转移方程(初始：`dp[0] = nums[0]`)：
  >
  > - 当`dp[i-1] > 0`：`dp[i] = dp[i-1] + nums[i];`
  > - 当`dp[i-1] <= 0`: `dp[i] = nums[i];`
  >
  > 空间复杂度降低：将原数组nums用作dp列表，即直接在nums上修改（也可以与上一题使用单独的一个变量来代替dp列表）
  >
  > ```java
  > public int maxSubArray(int[] nums) {
  >     int res = nums[0];
  >     for(int i = 0; i < nums.length; i++) {
  >         nums[i] += Math.max(nums[i-1], 0);
  >         res = Math.max(res, nums[i]);
  >     }
  >     return res;
  > }
  > ```

- [剑指 Offer 47. 礼物的最大价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/)

  > 空间复杂度降低：将原矩阵grid用作dp矩阵
  >
  > ```java
  > public int maxValue(int[][] grid) {
  >     int m = grid.length, n = grid[0].length;
  >     for(int j = 1; j < n; j++) // 初始化第一行
  >         grid[0][j] += grid[0][j - 1];
  >     for(int i = 1; i < m; i++) // 初始化第一列
  >         grid[i][0] += grid[i - 1][0];
  >     for(int i = 1; i < m; i++)
  >         for(int j = 1; j < n; j++) 
  >             grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
  >     return grid[m - 1][n - 1];
  > }
  > ```

- [剑指 Offer 46. 把数字翻译成字符串](https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

  > 状态定义：dp[i]代表以$x_i$结尾的数字的翻译方案数量
  >
  > 转移方程（采用**从右向左**遍历——整数求余的顺序，从`dp[n-2]`计算至`dp[0]`，初始：`dp[n-1]=1`, `dp[n-2] = 1`）：
  >
  > - 当可以余数$x_i$和余数$x_{i-1}$可以组合翻译时，`dp[i] = dp[i+1] + dp[i+2];`
  > - 否则，`dp[i] = dp[i+1];`
  >
  > 空间复杂度降低：用变量`a`表示`dp[i+1]`，变量b表示`dp[i+2]`，利用辅助变量`c`使`a`和`b`交替前进，用x，y用于记录求余。
  >
  > ```java
  > public int translateNum(int num) {
  >     int a = 1, b = 1, x, y = num % 10;
  >     while(num != 0) {
  >         num /= 10;
  >         x = num % 10;
  >         int tmp = 10 * x + y;
  >         int c = (tmp >= 10 && tmp <= 25) ? a + b : a;
  >         b = a;
  >         a = c;
  >         y = x;
  >     }
  >     return a;
  > }
  > ```

- [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode.cn/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

  > 状态定义：`dp[j]`代表以`c[j]`结尾的最长不重复子串的长度
  >
  > 转移方程：设`c[j]`左边距离最近的相同字符为`c[i]`，
  >
  > 1. 当`i < 0`，即`c[j]`左边无相同字符，则`dp[j] = dp[j-1] + 1;`
  > 2. 当`dp[j-1] < j - i`时，说明字符c[i]在子串`dp[j-1]`**区间之外**，则`dp[j] = dp[j-1] + 1;`
  > 3. 当`dp[j-1] >= j - i`时，说明字符c[i]在子串`dp[j-1]`**区间之内**，则`dp[j] = j - i;`
  >
  > 如何获取`i`，在遍历时使用哈希表保存并更新。
  >
  > ```java
  > public int lengthOfLongestSubstring(String s) {
  >     if(s.length() == 0) return 0;
  >     int res = 1, tmp = 1;
  >     char[] c = s.toCharArray();
  >     Map<Character, Integer> map = new HashMap<>();
  >     map.put(c[0], 0);
  >     for(int j = 1; j < c.length; j++) {
  >         if(map.get(c[j]) == null) {
  >             map.put(c[j], j);
  >             tmp = tmp + 1;
  >         } else {
  >             int i = map.get(c[j]);
  >             map.put(c[j], j);
  >             if(tmp < j - i) {
  >                 tmp = tmp + 1;
  >             } else {
  >                 tmp = j - i;
  >             }
  >         }
  >         res = Math.max(res, tmp);
  >     }
  >     return res;
  > }
  > ```
  >

- [剑指 Offer 19. 正则表达式匹配](https://leetcode.cn/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

  > 总体思路：从`s[:1]`和`p[:1]`是否能匹配开始判断，每轮添加一个字符并判断是否匹配，最终得到`s[:n]`是否能与`p[:m]`匹配；
  >
  > 下一轮的匹配状态有两种：
  >
  > - 添加一个字符$s_{i+1}$后是否能匹配？
  > - 添加一个字符$p_{i+1}$后是否能匹配？
  >
  > 状态定义：`dp[i][j]`代表字符串`s`的前`i`个字符和`p`的前`j`个字符能否匹配
  >
  > 转移方程（`dp[0][0]=true`代表空字符状态，`dp[i][j]`对应的添加字符是`s[i-1]`和`p[j-1]`）：
  >
  > - **当`p[j-1]='*'`时，`dp[i][j]`为true当满足以下任意情况：**
  >   1. **`dp[i][j-2]`，表示将字符组合`p[j-2]*`看作出现0次**
  >   2. **`dp[i-1][j] && s[i-1]==p[j-2]`，表示让`p[j-2]`多出现1次**
  >   3. **`dp[i-1][j] && p[j-2]=='.'`，表示让`'.'`多出现1次**
  > - **当`p[j-1]!='*'`时，`dp[i][j]`为true当满足以下任意情况：**
  >   1. **`dp[i-1][j-1] && s[i-1]==p[j-1]`，表示让字符`p[j-1]`多出现1次**
  >   2. **`dp[i-1][j-1] && p[j-1]=='.'`，即将字符`.`看作字符`s[i-1]`**
  >
  > dp矩阵首行初始化：`dp[0][0] = true`以及`dp[0][j] = dp[0][j-2]&&p[j-1]='*'`
  >
  > 【做题时状态的转移方程可以结合dp矩阵的填充过程进行总结】
  >
  > <img src="https://pic.leetcode-cn.com/1614516402-gBEUfu-Picture19.png" alt="Search in sidebar query" style="zoom:25%;" />
  >
  > ```java
  > public boolean isMatch(String s, String p) {
  >     int m = s.length() + 1, n = p.length() + 1;
  >     boolean[][] dp = new boolean[m][n];
  >     dp[0][0] = true;
  >     for(int j = 2; j < n; j += 2) {
  >         dp[0][j] = dp[0][j - 2] && p.charAt(j - 1) == '*';
  >     }
  >     for(int i = 1; i < m;i++){
  >         for(int j = 1; j < n; j++) {
  >             if(p.charAt(j-1) == '*') {
  >                 dp[i][j] = (dp[i][j-2]) || (dp[i-1][j] && s.charAt(i-1)==p.charAt(j-2)) || (dp[i-1][j] && p.charAt(j-2)=='.'); 
  >             } else {
  >                 dp[i][j] = (dp[i-1][j-1] && s.charAt(i-1) == p.charAt(j-1)) || (dp[i-1][j-1] && p.charAt(j-1) == '.');
  >             }
  >         }
  >     }
  >     return dp[m-1][n-1];
  > }
  > ```

- [剑指 Offer 49. 丑数](https://leetcode.cn/problems/chou-shu-lcof/)

  > 递推性质："丑数 = 某较小丑数 * 因子(2/3/5)"
  >
  > 状态定义：`dp[i]`代表第`i+1`个丑数，a,b,c分别表示不同因子下较小丑数的dp数组索引
  >
  > 转移方程（初始状态：`dp[0]=1`,`a=1,b=1,c=1`）：
  >
  > - `dp[i] = min(2*dp[a], 3*dp[b], 5*dp[c]);`
  > - 更新索引a,b,c的值，例如，上一步如果`dp[i]`是用`2*dp[a]`进行更新，则索引a++，其他同理
  >
  > ```java
  > public int nthUglyNumber(int n) {
  >     int a = 0, b = 0, c = 0;
  >     int[] dp = new int[n];
  >     dp[0] = 1;
  >     for(int i = 1; i < n; i++) {
  >         dp[i] = Math.min(Math.min(2*dp[a], 3*dp[b]), 5*dp[c]);
  >         if(dp[i] == 2*dp[a]) a++;
  >         if(dp[i] == 3*dp[b]) b++;
  >         if(dp[i] == 5*dp[c]) c++;
  >     }
  >     return dp[n-1];
  > }
  > ```

- [剑指 Offer 60. n个骰子的点数](https://leetcode.cn/problems/nge-tou-zi-de-dian-shu-lcof/)

  > 暴力法：$O(6^n)$
  >
  > 动态规划
  >
  > 状态定义：令输入n个骰子的解（概率列表）为f(n)，其中「点数和」`x`的概率为f(n, x)。
  >
  > 转移方程：
  > $$
  > f(n, x) = \sum^{6}_{i=1}f(n-1, x-i) \times \frac{1}{6}
  > $$
  > 越界问题：
  >
  > <img src="https://pic.leetcode-cn.com/1614960989-mMonMs-Picture3.png" alt="Picture3.png" style="zoom:25%;" />
  >
  > ```java
  > public double[] dicesProbability(int n) {
  >     double[] dp = new double[6];
  >     Arrays.fill(dp, 1.0 / 6.0);
  >     for(int i = 2; i <= n; i++) {
  >         double[] tmp = new double[5 * i + 1];
  >         for(int j = 0; j < dp.length; j++) {
  >             for(int k = 0; k < 6; k++) {
  >                 tmp[j + k] += dp[j] / 6.0;
  >             }
  >         }
  >         dp = tmp;
  >     }
  >     return dp;
  > }
  > ```