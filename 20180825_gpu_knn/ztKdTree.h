/*******************************************************************
 *	
 *	ZhengTu(Beijing) Laser Technology Co., Ltd
 *	(http://www.ztlidar.com/)
 *
 *	作者:	Sun Zhenxing
 *	创建日期:	20180825
 *
 *	说明：主要实现了多维kdtree的建立、最邻近搜索（个数、距离），
 *	      使用数组形式存储索引，没有编写删除、插入的功能，搜索
 *		  效率优于pcl
 *
 ******************************************************************/

#ifndef ZTKDTREE_H
#define ZTKDTREE_H

#include <vector>

namespace zt
{
#define SAMPLE_MEAN 1024

	class ZtKDTree
	{
	public:
		ZtKDTree();
		ZtKDTree(int dimension, unsigned int sz);
		~ZtKDTree();

		int setSize(int dimension, unsigned int sz);
		int setOffset(double oft[]);
		int setData(float *indata);	 // 数据已经减去偏移量
		int setData(double *indata); // 数据未减去偏移量
		int buildTree();

		// 查找最近点
		int findNearest(float *p);
		int findNearest(double *p);

		// 查找最近的k个点
		// 使用stl堆栈
		int findKNearestsSTL(float *p, int k, int *res);
		// 使用自编写堆栈 none third party, 比stl快5倍左右
		int findKNearestsNTP(float *p, int k, int *res, float *dit);
		int findKNearests(double *p, int k, int *res);

		// 查找给定范围内的点
		int findNearestRange(float *p, float range, std::vector<int> &res);
		int findNearestRange(double *p, float range, std::vector<int> &res);

		// gpu设备是否可用
		bool isGpuEnable;

		//// gpu查找最近的k个点
		//int gpuInit(int nn);
		//int gpuFindKNearests(float *p, int k, int *res);
		//int gpuFindKNearests(double *p, int k, int *res);

		int outKdTree(const char *outPath);

	private:
		int nDimension;
		unsigned int treeSize;
		double *offset;

		int treeRoot;	// 树根节点
		int **tree;		// 4 * n :分割维度、父节点、左子树、右子树
		int *treePtr;	// 使用一维数据表示二维数组，存储建立的kdtree索引

		/*
		*	所有数据存储在一维数组dataPtr里，data分别是x/y/z等数据的起始地址
		*	因此，建树及knn只需传递数据的索引编号即可
		*/
		float **data;
		float *dataPtr;

		// 建立kdtree
		int buildTree(int *indices, int count, int parent);
		
		/*
		*	建树功能函数
		*/
		int chooseSplitDimension(int *ids, int sz, float &key);
		int chooseMiddleNode(int *ids, int sz, int dim, float key);
		float computeDistance(float *p, int n);
	};
}

#endif