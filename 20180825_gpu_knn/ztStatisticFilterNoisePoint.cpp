#include "ztStatisticFilterNoisePoint.h"
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include "ztKdTree.h"


ZtStatisticFilterNoisePoint::ZtStatisticFilterNoisePoint()
	:mk(25), mt(1.0)
{

}

ZtStatisticFilterNoisePoint::~ZtStatisticFilterNoisePoint()
{

}

int ZtStatisticFilterNoisePoint::setParameter(int mean_k, float std_mul)
{
	mk = mean_k;
	mt = std_mul;

	return 0;
}

int ZtStatisticFilterNoisePoint::applyFilter(int n, float *pts, bool *res)
{
	if (!pts || !res || n < 1)
	{
		printf("Can't process null pointer!\n");
		return 1;
	}

	// 第一步：建立kdtree

	zt::ZtKDTree ztkdt;

	ztkdt.setSize(3, n);
	ztkdt.setData(pts);
	ztkdt.buildTree();


	// 第二步：查找临近点，计算距离均值

	memset(res, 1, n * sizeof(bool));	// 将res全部初始化为true

	float *distances = new float[n];	// 存储每一个点的平均距离

#pragma omp parallel for num_threads(omp_get_num_procs())
	for (int i = 0; i < n; i++)
	{
		int *knn = new int[mk];			// k临近索引
		float *knd = new float[mk];		// k临近距离

		ztkdt.findKNearestsNTP(pts + i * 3, mk, knn, knd);

		float sum = 0;
		for (int j = 0; j < mk; j++)
		{
			sum += knd[j];
		}

		distances[i] = sum / mk;

		delete[] knn;
		delete[] knd;
	}

	double sum = 0, sq_sum = 0;
	for (int i = 0; i < n; ++i)
	{
		sum += distances[i];
		sq_sum += distances[i] * distances[i];
	}

	double mean = sum / n;
	double variance = (sq_sum - sum * sum / n) / (n - 1);
	double stddev = sqrt(variance);

	float distance_threshold = mean + mt * stddev;

	// 第三步：根据距离阈值过滤点云

	for (int i = 0; i < n; i++)
	{
		if (distances[i] > distance_threshold)
		{
			res[i] = false;
		}
	}

	delete[] distances;

	return 0;
}

int ZtStatisticFilterNoisePoint::applyFilter_2(int n, float *pts, bool *res)
{
	// 一：先预处理进行点云分块

	// 1) 查找点云包围盒
	double lx = 9999999, ly = 9999999, rx = -9999999, ry = -9999999;

	for (int i = 0; i < n; i++)
	{
		lx = pts[i * 3 + 0] < lx ? pts[i * 3 + 0] : lx;
		ly = pts[i * 3 + 1] < ly ? pts[i * 3 + 1] : ly;

		rx = pts[i * 3 + 0] > rx ? pts[i * 3 + 0] : rx;
		ry = pts[i * 3 + 1] > ry ? pts[i * 3 + 1] : ry;
	}

	// 2) 根据包围盒的大小进行分块
	double ft = (rx - lx) / (ry - ly);

	// 水平、垂直分块
	int nhb = 0, nvb = 0;

	// 默认最多分8行或8列
	nhb = int(8 * ft / (ft + 1) + 0.5);
	nvb = int(8 * 1  / (ft + 1) + 0.5);

	// 防止临界点越界 + 1
	double hstep = (rx - lx + 1) / nhb;
	double vstep = (ry - ly + 1) / nvb;

// 	printf("%.3lf\t%.3lf\n%d\t%d\n%.3lf\t%.3lf\n",
// 		rx - lx, ry - ly, nhb, nvb, hstep, vstep);

	// 3) 对点云进行分块处理
	
	// 需先统计每一块的点数
	int *count = new int[nhb * nvb];
	for (int i = 0; i < nhb * nvb; i++)
	{
		count[i] = 0;
	}

	int c = 0, r = 0;
	for (int i = 0; i < n; i++)
	{
		c = int((pts[i * 3 + 0] - lx) / hstep) + int((pts[i * 3 + 1] - ly) / vstep) * nhb;
		count[c]++;
	}

	// 分配索引内存
	int **indices = new int *[nhb * nvb];
	for (int i = 0; i < nhb * nvb; i++)
	{
		indices[i] = new int[count[i]];
	}

	// 进行点云分块
	for (int i = 0; i < nhb * nvb; i++)
	{
		count[i] = 0;
	}

	for (int i = 0; i < n; i++)
	{
		c = int((pts[i * 3 + 0] - lx) / hstep) + int((pts[i * 3 + 1] - ly) / vstep) * nhb;
		indices[c][count[c]++] = i;
	}

	// 二：逐块处理，进行点云滤波
	for (int i = 0; i < nhb * nvb; i++)
	{
		if (count[i] < 1)
		{
			continue;
		}

		float *points = new float[count[i] * 3];
		for (int j = 0; j < count[i]; j++)
		{
			points[j * 3 + 0] = pts[indices[i][j] * 3 + 0];
			points[j * 3 + 1] = pts[indices[i][j] * 3 + 1];
			points[j * 3 + 2] = pts[indices[i][j] * 3 + 2];
		}

		bool *result = new bool[count[i]];

		if (applyFilter(count[i], points, result))
		{
			continue;
		}

		for (int j = 0; j < count[i]; j++)
		{
			res[indices[i][j]] = result[j];
		}

		delete[] points;
		delete[] result;
	}

	delete[] count;
	for (int i = nhb * nvb - 1; i >= 0; i--)
	{
		delete[] indices[i];
	}
	delete[] indices;

	return 0;
}