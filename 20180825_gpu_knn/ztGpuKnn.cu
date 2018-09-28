
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <string.h>
#include "ztGpuKnn.h"

/*********************** gpu指针定义 ***********************/
int   *_gpuKdTreeIndex;
float *_gpuKdTreeData;
float *_gpuSearchPoint;
int *_gpuSearchResult;

int *_gpuKNeighbors;
float *_gpuKDistances;
/***********************************************************/

int initCudaForKdtree(int n, int dim, int nn, int *index, float *treeData)
{
	if (cudaMalloc((void **)&_gpuKdTreeIndex, 4 * n * sizeof(int)) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMemcpy(_gpuKdTreeIndex, index, 4 * n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMalloc((void **)&_gpuKdTreeData, n * dim * sizeof(float)) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMemcpy(_gpuKdTreeData, treeData, n * dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMalloc((void **)&_gpuSearchPoint, ALLTHREADS * dim * sizeof(float)) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMalloc((void **)&_gpuSearchResult, ALLTHREADS * nn * sizeof(int)) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMalloc((void **)&_gpuKNeighbors, ALLTHREADS * nn * sizeof(int)) != cudaSuccess)
	{
		return 1;
	}

	if (cudaMalloc((void **)&_gpuKDistances, ALLTHREADS * nn * sizeof(float)) != cudaSuccess)
	{
		return 1;
	}

	return 0;
}

__global__ void gpuSearchKernal(int *gpuKN, float *gpuKD, int root, int *tree, int ndim, int size, float *data, int nn, float *points, int *res)
{
	int i = threadIdx.x + blockIdx.x * CUDA_THREAD;

	float p[3] = { points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2] };

	// 数组形式的最大堆
	//int *kNeighbors = gpuKN + i * nn;
	//float *kNDistance = gpuKD + i * nn;
	int kNeighbors[10];
	float kNDistance[10];
	int _currentNNode = 0;

	// 数组形式的路径堆栈
	int paths[64];
	int _currentPath = 0;

	// 记录查找路径
	int node = root;
	while (node > -1)
	{
		paths[_currentPath++] = node;

		node = p[tree[node]] <= data[tree[node] * size + node] ? tree[2 * size + node] : tree[3 * size + node];
	}

	kNeighbors[_currentNNode] = -1;
	kNDistance[_currentNNode++] = 9999999;

	// 回溯路径
	float distance = 0;
	while (_currentPath > 0)
	{
		node = paths[_currentPath-- - 1];

		float sum = 0;
		for (int j = 0; j < ndim; j++)
		{
			sum += (p[j] - data[j * size + node]) * (p[j] - data[j * size + node]);
		}
		distance = sum;

		if (_currentNNode < nn)
		{
			kNeighbors[_currentNNode] = node;
			kNDistance[_currentNNode++] = distance;

			// 当达到k个节点后，进行最大堆排序
			if (_currentNNode == nn)
			{
				for (int j = _currentNNode / 2 - 1; j >= 0; j--)
				{
					int parent = j;

					for (int son = j * 2 + 1; son <= _currentNNode; son = son * 2 + 1)
					{
						if (son + 1 < _currentNNode && kNDistance[son] < kNDistance[son + 1])
							son++;

						if (kNDistance[parent] < kNDistance[son])  // 如果父节点小于子节点，则交换
						{
							float tempD = kNDistance[parent];
							int tempI = kNeighbors[parent];
							kNDistance[parent] = kNDistance[son];
							kNeighbors[parent] = kNeighbors[son];
							kNDistance[son] = tempD;
							kNeighbors[son] = tempI;
						}

						parent = son;
					}
				}
			}
		}
		else
		{
			if (distance < kNDistance[0])
			{
				// pop
				kNeighbors[0] = kNeighbors[_currentNNode - 1];
				kNDistance[0] = kNDistance[_currentNNode - 1];

				// 删除堆顶后，要重构最大堆
				int parent = 0;
				int son = parent * 2 + 1;
				for (; son < _currentNNode - 1; son = son * 2 + 1)
				{
					if (son + 1 < _currentNNode - 1 && kNDistance[son] < kNDistance[son + 1])
						son++;

					if (kNDistance[parent] < kNDistance[son])  // 如果父节点小于子节点，则交换
					{
						float tempD = kNDistance[parent];
						int tempI = kNeighbors[parent];
						kNDistance[parent] = kNDistance[son];
						kNeighbors[parent] = kNeighbors[son];
						kNDistance[son] = tempD;
						kNeighbors[son] = tempI;
					}

					parent = son;
				}

				// push
				son = _currentNNode - 1;
				parent = (son - 1) / 2;
				while (son != 0 && distance > kNDistance[parent])
				{
					kNeighbors[son] = kNeighbors[parent];
					kNDistance[son] = kNDistance[parent];
					son = parent;
					parent = (son - 1) / 2;
				}

				kNDistance[son] = distance;
				kNeighbors[son] = node;
			}
		}

		if (tree[2 * size + node] + tree[3 * size + node] > -2)
		{
			int dim = tree[node];
			if (p[dim] > data[dim * size + node])
			{
				if (p[dim] - data[dim * size + node] < kNDistance[0] && tree[2 * size + node] > -1)
				{
					int reNode = tree[2 * size + node];
					while (reNode > -1)
					{
						paths[_currentPath++] = reNode;

						reNode = p[tree[reNode]] <= data[tree[reNode] * size + reNode] ? tree[2 * size + reNode] : tree[3 * size + reNode];
					}
				}
			}
			else
			{
				if (data[dim * size + node] - p[dim] < kNDistance[0] && tree[3 * size + node] > -1)
				{
					int reNode = tree[3 * size + node];
					while (reNode > -1)
					{
						paths[_currentPath++] = reNode;

						reNode = p[tree[reNode]] <= data[tree[reNode] * size + reNode] ? tree[2 * size + reNode] : tree[3 * size + reNode];
					}
				}
			}
		}
	}

	// 进行堆排序
	for (int j = _currentNNode - 1; j > 0; j--)
	{
		int tempI = kNeighbors[0];
		float tempD = kNDistance[0];
		kNeighbors[0] = kNeighbors[j];
		kNDistance[0] = kNDistance[j];
		kNeighbors[j] = tempI;
		kNDistance[j] = tempD;

		int parent = 0;
		int son = parent * 2 + 1;
		for (; son < j; son = parent * 2 + 1)
		{
			if (son + 1 < j && kNDistance[son] < kNDistance[son + 1])
				son++;

			if (kNDistance[parent] < kNDistance[son])
			{
				tempD = kNDistance[parent];
				tempI = kNeighbors[parent];
				kNDistance[parent] = kNDistance[son];
				kNeighbors[parent] = kNeighbors[son];
				kNDistance[son] = tempD;
				kNeighbors[son] = tempI;
			}

			parent = son;
		}
	}

	int j = nn;
	while (j != 0)
	{
		j--;
		res[i * nn + j] = kNeighbors[j];
	}
}


int gpuSearchKnnKdtree(int root, int ndim, int size, int nn, float *points, int *res)
{
	if (!points || cudaMemcpy(_gpuSearchPoint, points, ALLTHREADS * ndim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		return 1;
	}

	gpuSearchKernal<<<CUDA_BLOCK, CUDA_THREAD>>>(_gpuKNeighbors, _gpuKDistances, root, _gpuKdTreeIndex, 
		ndim, size, _gpuKdTreeData, nn, _gpuSearchPoint, _gpuSearchResult);

	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		return 1;
	}

	if (cudaMemcpy(res, _gpuSearchResult, ALLTHREADS * nn * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		return 1;
	}

	return 0;
}

int gpuFreeCuda()
{
	cudaFree(_gpuKdTreeIndex);
	cudaFree(_gpuKdTreeData);
	cudaFree(_gpuSearchPoint);
	cudaFree(_gpuSearchResult);

	cudaFree(_gpuKNeighbors);
	cudaFree(_gpuKDistances);

	return 0;
}

int getCudaDeviceCount()
{
	int count;
	cudaGetDeviceCount(&count);

	return count;
}

int getCudaDeviceNames(int i, char name[])
{
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
	{
		int n = strlen(prop.name);
		for (int i = 0; i < n; i++)
		{
			name[i] = prop.name[i];
		}
		name[n] = '\0';

		return 0;
	}

	return 1;
}

int setCudaStatus(int ndevice)
{
	if (cudaSetDevice(ndevice) == cudaSuccess)
	{
		return 0;
	}

	return 1;
}