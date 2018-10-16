/*******************************************************************
*
*	ZhengTu(Beijing) Laser Technology Co., Ltd
*	(http://www.ztlidar.com/)
*
*	作者:	Sun Zhenxing
*	创建日期:	20180923
*
*	说明：实现点云的统计滤波去噪功能，有两个滤波接口，第一个是直接将
*	      点云加进去处理，第二个是进行内部分块后，再进行滤波处理，目
*		  的是防止点云分布极为不均匀时，导致滤波错误
*
******************************************************************/

#ifndef ZTSTATISTICFILTERNOISEPOINT_H
#define ZTSTATISTICFILTERNOISEPOINT_H


class ZtStatisticFilterNoisePoint
{
public:
	ZtStatisticFilterNoisePoint();
	~ZtStatisticFilterNoisePoint();

	int setParameter(int mean_k, float std_mul);
	int applyFilter(int n, float *pts, bool *res);

	// 预处理分块后再滤波
	int applyFilter_2(int n, float *pts, bool *res);

private:
	int mk;		// 临近点数
	float mt;	// sigma倍数
};

#endif