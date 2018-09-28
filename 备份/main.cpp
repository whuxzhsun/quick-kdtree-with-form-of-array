#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <liblas\liblas.hpp>
#include "ztKdTree.h"

using namespace std;

int testKnn();

int main()
{
	testKnn();

	return 0;
}

int testKnn()
{
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
/*	n = 100000;*/

	unsigned int pointCount = 0;

	for (int j = 0; j < 1/*m*/; j++)
	{
		clock_t start = clock();

		float *pts = new float[n * 3];

		for (int k = 0; k < n; k++)
		{
			reader.ReadNextPoint();
			inPt = reader.GetPoint();

			pts[k * 3 + 0] = inPt.GetX() - inHeader.GetOffsetX();
			pts[k * 3 + 1] = inPt.GetY() - inHeader.GetOffsetY();
			pts[k * 3 + 2] = inPt.GetZ() - inHeader.GetOffsetZ();
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

// 		int snode = 2262;
// 		do
// 		{
// 			int pnt = kdt.tree[1][snode];
// 			char lr;
// 			int bn = 0;
// 
// 			if (kdt.tree[2][pnt] == snode)
// 			{
// 				lr = 'r';
// 				bn = kdt.tree[3][pnt];
// 			}
// 			else
// 			{
// 				{
// 					lr = 'l';
// 					bn = kdt.tree[2][pnt];
// 				}
// 			}
// 
// 			double dx, dy, dz;
// 			dx = kdt.data[0][snode] - pts[100 * 3 + 0];
// 			dy = kdt.data[1][snode] - pts[100 * 3 + 1];
// 			dz = kdt.data[2][snode] - pts[100 * 3 + 2];
// 
// 			fprintf_s(fp, "%6d\t%d\t%c\t%d\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\t%7.3f\n",
// 				snode, kdt.tree[0][pnt], lr, bn, 
// 				kdt.data[0][snode], kdt.data[1][snode], kdt.data[2][snode],
// 				dx,
// 				dy,
// 				dz, sqrt(dx * dx + dy * dy + dz * dz));
// 
// 			snode = pnt;	// Ë÷Òý¸¸½Úµã
// 		} while (snode != 0);

		for (int i = 0; i < n; i += 100)
		{
			float sPt[3] = { pts[i * 3 + 0],
				pts[i * 3 + 1],
				pts[i * 3 + 2]};
			/*int nst = kdt.findNearest(sPt);*/

			int knn[10];
			kdt.findKNearests2(sPt, 10, knn);

			/*fprintf_s(fp, "%-6d\t%-6d\n", i, nst);*/
			fprintf_s(fp, "%-6d\t%7.3f\t%7.3f\n", i, sPt[0], sPt[1]);
			for (int k = 0; k < 10; k++)
			{
				double dx, dy, dz;
				dx = sPt[0] - pts[knn[k] * 3 + 0];
				dy = sPt[1] - pts[knn[k] * 3 + 1];
				dz = sPt[2] - pts[knn[k] * 3 + 2];
				fprintf_s(fp, "%-6d\t%5.3f\t", knn[k], 
					sqrt(dx * dx + dy * dy + dz * dz));
			}
			fprintf_s(fp, "\n");
		}
		fclose(fp);

		printf("\tcost time of knn: %.3f\n", (clock() - start) / 1000.0);

		delete[] pts;	
	}

	return 0;
}