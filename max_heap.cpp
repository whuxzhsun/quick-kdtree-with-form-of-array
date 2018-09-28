
void buildMaxHeap(int arr[], int n)
{
    for (int i = n / 2 - 1; i >= 0; i--)
    {
        for (int son = i * 2; son <= n; son *= 2)
        {
            if (son + 1 < n && arr[son] < arr[son + 1])
                son++;

            if (arr[i] < arr[son])  // 如果父节点小于子节点，则交换
            {
                int temp = arr[i];
                arr[i] = arr[son];
                arr[son] = temp;
            }
        }
    }
}

// 删除堆顶，将最后一个元素移到堆顶，并且将最后一个元素置为空
void pop(int arr[], int n)
{
    arr[0] = arr[n - 1];
    arr[n - 1] = -1;
}

// 遍历堆，寻找合适的插入位置
void push(int arr[], int n, int x)
{
    int i = n - 1;
    while (i != 0 && x > arr[i / 2])
    {
        arr[i] = arr[i / 2];
        i /= 2;
    }

    arr[i] = x;
}