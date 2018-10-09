#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <liblas\liblas.hpp>
#include "ztKdTree.h"
#include "ztGpuKnn.h"
#include <omp.h>
#include <string.h>
#include "ztStatisticFilterNoisePoint.h"

using namespace std;

int testKnn();
int testGpuKnn();
int testFilter();

int main()
{
	//	testKnn();
	testFilter();

	//	double sum = 0;

	// 	clock_t t = clock();
	// 
	// #pragma omp parallel for num_threads(omp_get_num_procs()), reduction(+:sum)
	// 	for (int j = 0; j < 10000000; j++)
	// 	{
	// 		sum += j;
	// 
	// 		/*printf("\n%d\n", sum);*/
	// 	}
	// 
	// 	printf("\tcost time of computing: %.3f\n", (clock() - t) / 1000.0);

	// 	bool *test = new bool[8];
	// 
	// 	memset(test, 1, 8 * sizeof(bool));
	// 
	// 	for (int i = 0; i < 8; i++)
	// 	{
	// 		if (test[i])
	// 		{
	// 			printf("%d\n", i);
	// 		}
	// 	}
	// 
	// 	delete[] test;

	return 0;
}

int testKnn()
{
	printf("Test CPU:\n");
	std::string inFile("D:\\20180419_R1000_ariborne_test_mta\\180419_024813_0.las");

	std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "can't open " << inFile << endl;

		return 1;
	}

	liblas::Reader reader(ifs);
	liblas::Header inHeader = reader.GetHeader();

	liblas::Point inPt(&inHeader);

	int m = inHeader.GetPointRecordsCount() / 1000000 + 1;
	int n = inHeader.GetPointRecordsCount() / m;
	n = (n / ALLTHREADS) * ALLTHREADS;
	printf("number of points: %d\n", n);

	unsigned int pointCount = 0;

	for (int j = 0; j < 1/*m*/; j++)
	{
		clock_t start = clock();

		float *pts = new float[n * 3];

		for (int k = 0; k < n; k++)
		{
			reader.ReadNextPoint();
			inPt = reader.GetPoint();

			pts[k * 3 + 0] = float(inPt.GetX() - inHeader.GetOffsetX());
			pts[k * 3 + 1] = float(inPt.GetY() - inHeader.GetOffsetY());
			pts[k * 3 + 2] = float(inPt.GetZ() - inHeader.GetOffsetZ());
		}

		clock_t end = clock();

		printf("\tcost time of reading points: %.3f\n", (end - start) / 1000.0);

		start = clock();

		zt::ZtKDTree kdt;
		kdt.setSize(3, n);
		kdt.setData(pts);
		kdt.buildTree();

		end = clock();

		printf("\tcost time of bulid kdtree: %.3f\n", (end - start) / 1000.0);

		start = clock();

		FILE *fp;
		fopen_s(&fp, "D:\\20180419_R1000_ariborne_test_mta\\index_me.txt", "w");
		if (!fp)
		{
			printf("Can.t create file!");
			return 1;
		}

		printf("number of core = %d\n", omp_get_num_procs());
#pragma omp parallel for num_threads(omp_get_num_procs()/* / 2*/)
		for (int i = 0; i < n; i += 1)
		{
			float sPt[3] = { pts[i * 3 + 0],
				pts[i * 3 + 1],
				pts[i * 3 + 2] };
			/*int nst = kdt.findNearest(sPt);*/

			int knn[50];
			float dist[50];
			kdt.findKNearestsNTP(sPt, 50, knn, dist);

			/*fprintf_s(fp, "%-6d\t%-6d\n", i, nst);*/
			// 			fprintf_s(fp, "%d\n", i);
			// 			for (int k = 0; k < 10; k++)
			// 			{
			// 				double dx, dy, dz;
			// 				dx = sPt[0] - pts[knn[k] * 3 + 0];
			// 				dy = sPt[1] - pts[knn[k] * 3 + 1];
			// 				dz = sPt[2] - pts[knn[k] * 3 + 2];
			// 				fprintf_s(fp, "%-6d\t%5.3f\t", knn[k], 
			// 					sqrt(dx * dx + dy * dy + dz * dz));
			// 			}
			// 			fprintf_s(fp, "\n");
		}
		fclose(fp);

		printf("\tcost time of knn: %.3f\n", (clock() - start) / 1000.0);

		delete[] pts;
	}

	return 0;
}

int testGpuKnn()
{
	/*	std::string inFile("D:\\20180419_R1000_ariborne_test_mta\\180419_024813_0.las");

		std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
		if (!ifs.is_open())
		{
		std::cout << "can't open " << inFile << endl;

		return 1;
		}

		liblas::Reader reader(ifs);
		liblas::Header inHeader = reader.GetHeader();

		liblas::Point inPt(&inHeader);

		int m = inHeader.GetPointRecordsCount() / 1000000 + 1;
		int n = inHeader.GetPointRecordsCount() / m;

		int jump = n / ALLTHREADS;
		n = (n / ALLTHREADS) * ALLTHREADS;
		printf("number of points: %d\n", n);

		unsigned int pointCount = 0;

		for (int j = 0; j < 1; j++)
		{
		clock_t start = clock();

		float *pts = new float[n * 3];

		for (int k = 0; k < n; k++)
		{
		reader.ReadNextPoint();
		inPt = reader.GetPoint();

		pts[k * 3 + 0] = float(inPt.GetX() - inHeader.GetOffsetX());
		pts[k * 3 + 1] = float(inPt.GetY() - inHeader.GetOffsetY());
		pts[k * 3 + 2] = float(inPt.GetZ() - inHeader.GetOffsetZ());
		}

		clock_t end = clock();

		printf("\tcost time of reading points: %.3f\n", (end - start) / 1000.0);

		start = clock();

		zt::ZtKDTree kdt;
		kdt.setSize(3, n);
		kdt.setData(pts);
		kdt.buildTree();

		end = clock();

		printf("\tcost time of bulid kdtree: %.3f\n", (end - start) / 1000.0);

		start = clock();



		FILE *fp;
		fopen_s(&fp, "D:\\20180419_R1000_ariborne_test_mta\\index_me_gpu.txt", "w");
		if (!fp)
		{
		printf("Can't create file!");
		return 1;
		}

		clock_t t1 = clock();

		int nn = 10;
		if (kdt.gpuInit(nn))
		{
		printf("Failed to init gpu!\n");
		return 1;
		}

		printf("cost time of init gpu: %.3f\n", (clock() - t1) / 1000.0);

		for (int i = 0; i < jump; i += 1)
		{
		float *sPts = new float[ALLTHREADS * 3];
		int *res = new int[ALLTHREADS * nn];

		for (int j = 0; j < ALLTHREADS; j++)
		{
		sPts[j * 3 + 0] = pts[(j * jump + i) * 3 + 0];
		sPts[j * 3 + 1] = pts[(j * jump + i) * 3 + 1];
		sPts[j * 3 + 2] = pts[(j * jump + i) * 3 + 2];
		}

		if (kdt.gpuFindKNearests(sPts, nn, res))
		{
		printf("Failed to use gpu!\n");
		return 1;
		}

		for (int j = 0; j < ALLTHREADS; j++)
		{
		fprintf_s(fp, "%d\n", j * jump + i);
		for (int k = 0; k < nn; k++)
		{
		double dx, dy, dz;
		dx = sPts[j * 3 + 0] - pts[res[j * nn + k] * 3 + 0];
		dy = sPts[j * 3 + 1] - pts[res[j * nn + k] * 3 + 1];
		dz = sPts[j * 3 + 2] - pts[res[j * nn + k] * 3 + 2];
		fprintf_s(fp, "%-6d\t%5.3f\t", res[j * nn + k],
		sqrt(dx * dx + dy * dy + dz * dz));
		}
		fprintf_s(fp, "\n");
		}

		delete[] res;
		delete[] sPts;
		}
		fclose(fp);

		printf("\tcost time of knn: %.3f\n", (clock() - start) / 1000.0);

		delete[] pts;
		}
		//*/
	return 0;
}

int testFilter()
{
	std::string inFile("D:\\20180419_R1000_ariborne_test_mta\\180419_024813_0.las");
	std::string outFile("D:\\20180419_R1000_ariborne_test_mta\\180419_024813_0_Filter.las");

	std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "can't open " << inFile << endl;

		return 1;
	}

	liblas::Reader reader(ifs);
	liblas::Header inHeader = reader.GetHeader();

	liblas::Point inPt(&inHeader);

	int m = inHeader.GetPointRecordsCount() / 1000000 + 1;
	int n = inHeader.GetPointRecordsCount() / m;

	printf("number of points: %d\n", m * n);

	std::ofstream ofs(outFile, std::ios::out | std::ios::binary);
	if (!ofs.is_open())
	{
		std::cout << "can't open " << outFile << endl;
		return 1;
	}

	liblas::Header outHeader;
	outHeader.SetVersionMajor(1);
	outHeader.SetVersionMinor(2);
	outHeader.SetDataFormatId(liblas::ePointFormat3);
	outHeader.SetScale(0.001, 0.001, 0.001);
	outHeader.SetOffset(inHeader.GetOffsetX(), inHeader.GetOffsetY(), inHeader.GetOffsetZ());

	liblas::Writer writer(ofs, outHeader);
	liblas::Point outPt(&outHeader);

	unsigned int pointCount = 0;
	unsigned int filterCount = 0;

	clock_t allStart = clock();
	for (int j = 0; j < m; j++)
	{
		clock_t start = clock();

		float *pts = new float[n * 3];
		bool *res = new bool[n];

		std::vector<liblas::Point> vpts;
		vpts.reserve(n);

		for (int k = 0; k < n; k++)
		{
			reader.ReadNextPoint();
			inPt = reader.GetPoint();
			vpts.push_back(inPt);

			pts[k * 3 + 0] = float(inPt.GetX() - inHeader.GetOffsetX());
			pts[k * 3 + 1] = float(inPt.GetY() - inHeader.GetOffsetY());
			pts[k * 3 + 2] = float(inPt.GetZ() - inHeader.GetOffsetZ());
		}

		clock_t end = clock();

		printf("\ncost time of reading points: %.3f\n", (end - start) / 1000.0);

		start = clock();

		ZtStatisticFilterNoisePoint fnp;
		fnp.setParameter(25, 1.0);
		fnp.applyFilter_2(n, pts, res);

		end = clock();

		printf("\tcost time of filtering point: %.3f\n", (end - start) / 1000.0);

		start = clock();

		for (int i = 0; i < n; i++)
		{
			if (!res[i])
			{
				filterCount++;
				outPt.SetClassification(7);
			}
			else
			{
				outPt.SetClassification(1);
			}

			pointCount++;

			/*outPt = vpts[i];*/

			outPt.SetX(vpts[i].GetX());
			outPt.SetY(vpts[i].GetY());
			outPt.SetZ(vpts[i].GetZ());

			outPt.SetIntensity(vpts[i].GetIntensity());

			writer.WritePoint(outPt);
		}		

		printf("\tcost time of writing point: %.3f\n", (clock() - start) / 1000.0);

		delete[] pts;
	}

	outHeader.SetMax(inHeader.GetMaxX(), inHeader.GetMaxY(), inHeader.GetMaxZ());
	outHeader.SetMin(inHeader.GetMinX(), inHeader.GetMinY(), inHeader.GetMinZ());
	outHeader.SetPointRecordsCount(pointCount);

	writer.SetHeader(outHeader);
	writer.WriteHeader();

	printf("\nreserve points : %d, remove points : %d\n", pointCount, filterCount);
	printf("\ncost time of filtering all points: %.3f\n", (clock() - allStart) / 1000.0);

	return 0;
}