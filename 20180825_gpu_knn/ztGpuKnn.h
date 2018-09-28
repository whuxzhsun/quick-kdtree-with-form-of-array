#ifndef ZTGPUKNN_H
#define ZTGPUKNN_H

#define CUDA_BLOCK	64
#define CUDA_THREAD 32
#define ALLTHREADS	(CUDA_BLOCK * CUDA_THREAD)

/*
**	函数功能：预先为gpu运算分配内存，并将kdtree数据copy到gpu中
**	参数定义：	n-点数， dim-维度， nn-临近搜索点数， index-kdtree索引
**				treeData-点云数据
*/
int initCudaForKdtree(int n, int dim, int nn, int *index, float *treeData);

/*
**	函数功能：gpu加速搜索临近点
**	参数定义：	root-树根， ndim-维度， size-树点数， nn-临近搜索点数
**				points-要搜索的点， res-搜索的结果
*/
int gpuSearchKnnKdtree(int root, int ndim, int size, int nn, float *points, int *res);

/*
**	函数功能：释放cuda内存
*/
int gpuFreeCuda();

/*
**	函数功能：获取可用cuda设备个数
*/
int getCudaDeviceCount();

/*
**	函数功能：获取第i个设备的名称
**	参数定义：i-设备序列，name-设备名称
*/
int getCudaDeviceNames(int i, char name[]);

/*
**	函数功能：设定要使用的设备
**	参数定义：ndevice-设备的序列
*/
int setCudaStatus(int ndevice);

#endif