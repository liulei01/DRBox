#include <math.h>
#include <algorithm>
#include <iostream>
using namespace std;

struct Line
{
	int crossnum;//0:ignore; -1:all inner point; 2:two crossing point; 1:one crossing point
	int p1;//index of the start point
	int p2;//index of the end point
	int d[2][2];//the index of the start point after division
	double length;//the length after division
};

void OverlapSub (double *rbox1, double *rbox2, double *area)
{
    double xcenter1 = rbox1[0];
    double ycenter1 = rbox1[1];
    double width1 = rbox1[2];
    double height1 = rbox1[3];
    double angle1 = rbox1[4];
    double xcenter2 = rbox2[0];
    double ycenter2 = rbox2[1];
    double width2 = rbox2[2];
    double height2 = rbox2[3];
    double angle2 = rbox2[4];
    //for(int i=0;i<5;i++) cout<<rbox1[i]<<" ";
    //cout<<endl;
    //for(int i=0;i<5;i++) cout<<rbox2[i]<<" ";
    //cout<<endl;
	angle1 = -angle1;
	angle2 = -angle2;
	double angled = angle2 - angle1;
	angled *= (double)3.14159265/180;
	angle1 *= (double)3.14159265/180;
	//ofstream fout("Output.txt");
	area[0] = 0;
	double hw1 = width1 / 2;
	double hh1 = height1 /2;
	double hw2 = width2 / 2;
	double hh2 = height2 /2;
	double xcenterd = xcenter2 - xcenter1;
	double ycenterd = ycenter2 - ycenter1;
	double tmp = xcenterd * cosf(angle1) + ycenterd * sinf(angle1);
	ycenterd = -xcenterd * sinf(angle1) + ycenterd * cosf(angle1);
	xcenterd = tmp;
	double max_width_height1 = width1 > height1? width1 : height1;
	double max_width_height2 = width2 > height2? width2 : height2;
	if (sqrt(xcenterd * xcenterd + ycenterd * ycenterd) > 
		(max_width_height1 + max_width_height2) * 1.414214/2)
	{
		area[0] = 0;
		//fout<<endl<<"AREA = 0"<<endl;
		//fout.close();
		return;
	}
	if (fabs(sin(angled)) < 1e-3)
	{
		if (fabs(xcenterd) > (hw1 + hw2) || fabs(ycenterd) > (hh1 + hh2))
		{
			area[0] = 0;
			//fout<<endl<<"AREA = 0"<<endl;
			//fout.close();
			return;
		}
		else
		{
			double x_min_inter = -hw1 > (xcenterd - hw2)? -hw1 : (xcenterd - hw2);
			double x_max_inter = hw1 < (xcenterd + hw2)? hw1 : (xcenterd + hw2);
			double y_min_inter = -hh1 > (ycenterd - hh2)? -hh1 : (ycenterd - hh2);
			double y_max_inter = hh1 < (ycenterd + hh2)? hh1 : (ycenterd + hh2);
			const double inter_width = x_max_inter - x_min_inter;
			const double inter_height = y_max_inter - y_min_inter;
			const double inter_size = inter_width * inter_height;
			area[0] = inter_size;
            area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
			//LOG(INFO)<<"AREA = "<<area;
			//fout.close();
			return;
		}
	}
	if (fabs(cos(angled)) < 1e-3)
	{
		double x_min_inter = -hw1 > (xcenterd - hh2)? -hw1 : (xcenterd - hh2);
		double x_max_inter = hw1 < (xcenterd + hh2)? hw1 : (xcenterd + hh2);
		double y_min_inter = -hh1 > (ycenterd - hw2)? -hh1 : (ycenterd - hw2);
		double y_max_inter = hh1 < (ycenterd + hw2)? hh1 : (ycenterd + hw2);
		const double inter_width = x_max_inter - x_min_inter;
		const double inter_height = y_max_inter - y_min_inter;
		const double inter_size = inter_width * inter_height;
		area[0] = inter_size;
        area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
		//fout<<endl<<"AREA = "<<area<<endl;
		//fout.close();
		return;
	}
	
	double cos_angled = cosf(angled);
	double sin_angled = sinf(angled);
	double cos_angled_hw1 = cos_angled * hw1;
	double sin_angled_hw1 = sin_angled * hw1;
	double cos_angled_hh1 = cos_angled * hh1;
	double sin_angled_hh1 = sin_angled * hh1;
	double cos_angled_hw2 = cos_angled * hw2;
	double sin_angled_hw2 = sin_angled * hw2;
	double cos_angled_hh2 = cos_angled * hh2;
	double sin_angled_hh2 = sin_angled * hh2;
	
	// point20: (w/2, h/2)
	double point2x[4], point2y[4];
	point2x[0] = xcenterd + cos_angled_hw2 - sin_angled_hh2;
	point2y[0] = ycenterd + sin_angled_hw2 + cos_angled_hh2;
	// point21: (-w/2, h/2)
	point2x[1] = xcenterd - cos_angled_hw2 - sin_angled_hh2;
	point2y[1] = ycenterd - sin_angled_hw2 + cos_angled_hh2; 
	// point22: (-w/2, -h/2)
	point2x[2] = xcenterd - cos_angled_hw2 + sin_angled_hh2;
	point2y[2] = ycenterd - sin_angled_hw2 - cos_angled_hh2;
	// point23: (w/2, -h/2)
	point2x[3] = xcenterd + cos_angled_hw2 + sin_angled_hh2;
	point2y[3] = ycenterd + sin_angled_hw2 - cos_angled_hh2; 

	double pcenter_x = 0, pcenter_y = 0;
	int count = 0;

	// determine the inner point
	bool inner_side2[4][4], inner2[4];
	for(int i = 0; i < 4; i++)
	{
		inner_side2[i][0] = point2y[i] < hh1;
		inner_side2[i][1] = point2x[i] > -hw1;
		inner_side2[i][2] = point2y[i] > -hh1;
		inner_side2[i][3] = point2x[i] < hw1;
		inner2[i] = inner_side2[i][0] & inner_side2[i][1] & inner_side2[i][2] & inner_side2[i][3];
		if (inner2[i]) { pcenter_x += point2x[i]; pcenter_y += point2y[i]; count++;}
	}

	//similar operating for rbox1: angled -> -angled, xcenterd -> -xcenterd, ycenterd -> -ycenterd
	// point10: (w/2, h/2)
	double xcenterd_hat = - xcenterd * cos_angled - ycenterd * sin_angled;
	double ycenterd_hat = xcenterd * sin_angled - ycenterd * cos_angled;
	double point1x[4], point1y[4];
	
	point1x[0] = xcenterd_hat + cos_angled_hw1 + sin_angled_hh1;
	point1y[0] = ycenterd_hat - sin_angled_hw1 + cos_angled_hh1;
	// point21: (-w/2, h/2)
	point1x[1] = xcenterd_hat - cos_angled_hw1 + sin_angled_hh1;
	point1y[1] = ycenterd_hat + sin_angled_hw1 + cos_angled_hh1; 
	// point22: (-w/2, -h/2)
	point1x[2] = xcenterd_hat - cos_angled_hw1 - sin_angled_hh1;
	point1y[2] = ycenterd_hat + sin_angled_hw1 - cos_angled_hh1;
	// point23: (w/2, -h/2)
	point1x[3] = xcenterd_hat + cos_angled_hw1 - sin_angled_hh1;
	point1y[3] = ycenterd_hat - sin_angled_hw1 - cos_angled_hh1;
	
	// determine the inner point
	// determine the inner point
	bool inner_side1[4][4], inner1[4];
	for(int i = 0; i < 4; i++)
	{
		inner_side1[i][0] = point1y[i] < hh2;
		inner_side1[i][1] = point1x[i] > -hw2;
		inner_side1[i][2] = point1y[i] > -hh2;
		inner_side1[i][3] = point1x[i] < hw2;
		inner1[i] = inner_side1[i][0] & inner_side1[i][1] & inner_side1[i][2] & inner_side1[i][3];
	}
	point1x[0] = hw1;
	point1y[0] = hh1;
	// point21: (-w/2, h/2)
	point1x[1] = -hw1;
	point1y[1] = hh1;
	// point22: (-w/2, -h/2)
	point1x[2] = -hw1;
	point1y[2] = -hh1;
	// point23: (w/2, -h/2)
	point1x[3] = hw1;
	point1y[3] = -hh1;
	if (inner1[0]) { pcenter_x += hw1; pcenter_y += hh1; count++;}
	if (inner1[1]) { pcenter_x -= hw1; pcenter_y += hh1; count++;}
	if (inner1[2]) { pcenter_x -= hw1; pcenter_y -= hh1; count++;}
	if (inner1[3]) { pcenter_x += hw1; pcenter_y -= hh1; count++;}
	//find cross_points
	Line line1[4], line2[4];
	line1[0].p1 = 0; line1[0].p2 = 1;
	line1[1].p1 = 1; line1[1].p2 = 2;
	line1[2].p1 = 2; line1[2].p2 = 3;
	line1[3].p1 = 3; line1[3].p2 = 0;
	line2[0].p1 = 0; line2[0].p2 = 1;
	line2[1].p1 = 1; line2[1].p2 = 2;
	line2[2].p1 = 2; line2[2].p2 = 3;
	line2[3].p1 = 3; line2[3].p2 = 0;
	double pointc_x[4][4], pointc_y[4][4];
	for (int i = 0; i < 4; i++)
	{
		int index1 = line1[i].p1;
		int index2 = line1[i].p2;
		line1[i].crossnum = 0;
		if (inner1[index1] && inner1[index2])
		{
			if (i == 0 || i == 2) line1[i].length = width1;
			else line1[i].length = height1;
			line1[i].crossnum = -1;
			continue;
		}
		if (inner1[index1])
		{
			line1[i].crossnum ++;
			line1[i].d[0][0] = index1;
			line1[i].d[0][1] = -1;
			continue;
		}
		if (inner1[index2])
		{
			line1[i].crossnum ++;
			line1[i].d[0][0] = index2;
			line1[i].d[0][1] = -1;
			continue;
		}
	}
	for (int i = 0; i < 4; i++)
	{
		int index1 = line2[i].p1;
		double x1 = point2x[index1];
		double y1 = point2y[index1];
		int index2 = line2[i].p2;
		double x2 = point2x[index2];
		double y2 = point2y[index2];
		line2[i].crossnum = 0;
		if (inner2[index1] && inner2[index2])
		{
			if (i == 0 || i == 2) line2[i].length = width2;
			else line2[i].length = height1;
			line2[i].crossnum = -1;
			continue;
		}
		if (inner2[index1])
		{
			line2[i].crossnum ++;
			line2[i].d[0][0] = index1;
			line2[i].d[0][1] = -1;
		}
		else if (inner2[index2])
		{
			line2[i].crossnum ++;
			line2[i].d[0][0] = index2;
			line2[i].d[0][1] = -1;
		}
		double tmp1 = (y1*x2 - y2*x1) / (y1 - y2);
		double tmp2 = (x1 - x2) / (y1 - y2);
        //cout<<"tmp"<<" "<<i<<" "<<tmp1<<" "<<tmp2<<endl;
        //cout<<"x1="<<x1<<" x2="<<x2<<endl;
        //cout<<"y1="<<y1<<" y2="<<y2<<endl;
		double tmp3 = (x1*y2 - x2*y1) / (x1 - x2);
		double tmp4 = 1/tmp2 * hw1;
		tmp2 *= hh1;
		for (int j = 0; j < 4; j++)
		{
			int index3 = line1[j].p1;
			int index4 = line1[j].p2;
			if ((inner_side2[index1][j] != inner_side2[index2][j]) 
				&& (inner_side1[index3][i] != inner_side1[index4][i]))
			{
				switch (j)
				{
				case 0: 
					pointc_x[i][j] = tmp1 + tmp2;
					pointc_y[i][j] = hh1;
					break;
				case 1:
					pointc_y[i][j] = tmp3 - tmp4;
					pointc_x[i][j] = -hw1;
					break;
				case 2:
					pointc_x[i][j] = tmp1 - tmp2;
					pointc_y[i][j] = -hh1;
					break;
				case 3:
					pointc_y[i][j] = tmp3 + tmp4;
					pointc_x[i][j] = hw1;
					break;
				default:
					break;
				}
				line1[j].d[line1[j].crossnum][0] = i;
				line1[j].d[line1[j].crossnum ++][1] = j;
				line2[i].d[line2[i].crossnum][0] = i;
				line2[i].d[line2[i].crossnum ++][1] = j;
				pcenter_x += pointc_x[i][j];
				pcenter_y += pointc_y[i][j];
				count ++;
                //cout<<"pointc:"<<i<<" "<<j<<" "<<pointc_x[i][j]<<" "<<pointc_y[i][j]<<endl;
			}
		}
	}
	pcenter_x /= (double)count;
	pcenter_y /= (double)count;
	double pcenter_x_hat, pcenter_y_hat;
	pcenter_x_hat = pcenter_x - xcenterd;
	pcenter_y_hat = pcenter_y - ycenterd;
	tmp = cos_angled * pcenter_x_hat + sin_angled * pcenter_y_hat;
	pcenter_y_hat = -sin_angled * pcenter_x_hat + cos_angled * pcenter_y_hat;
	pcenter_x_hat = tmp;

	for (int i = 0; i < 4; i++)
	{
		if (line1[i].crossnum > 0)
		{
			if (line1[i].d[0][1] == -1)
			{
				if (i==0 || i==2) 
					line1[i].length = fabs(point1x[line1[i].d[0][0]] - pointc_x[line1[i].d[1][0]][line1[i].d[1][1]]);
				else
					line1[i].length = fabs(point1y[line1[i].d[0][0]] - pointc_y[line1[i].d[1][0]][line1[i].d[1][1]]);
			}
			else
			{
				if (i==0 || i==2)
					line1[i].length = fabs(pointc_x[line1[i].d[0][0]][line1[i].d[0][1]] - pointc_x[line1[i].d[1][0]][line1[i].d[1][1]]);
				else
					line1[i].length = fabs(pointc_y[line1[i].d[0][0]][line1[i].d[0][1]] - pointc_y[line1[i].d[1][0]][line1[i].d[1][1]]);
			}
		}
		if (line2[i].crossnum >0)
		{
			if (line2[i].d[0][1] == -1)
				line2[i].length = fabs(point2x[line2[i].d[0][0]] - pointc_x[line2[i].d[1][0]][line2[i].d[1][1]]);
			else
				line2[i].length = fabs(pointc_x[line2[i].d[0][0]][line2[i].d[0][1]] - pointc_x[line2[i].d[1][0]][line2[i].d[1][1]]);
			if(i == 0 || i == 2) line2[i].length *= width2 / fabs(point2x[line2[i].p1] - point2x[line2[i].p2]);
			else line2[i].length *= height2 / fabs(point2x[line2[i].p1] - point2x[line2[i].p2]);
		}
	}

	double dis1[4], dis2[4];
	dis1[0] = fabs(pcenter_y - hh1);
	dis1[1] = fabs(pcenter_x + hw1);
	dis1[2] = fabs(pcenter_y + hh1);
	dis1[3] = fabs(pcenter_x - hw1);
	dis2[0] = fabs(pcenter_y_hat - hh2);
	dis2[1] = fabs(pcenter_x_hat + hw2);
	dis2[2] = fabs(pcenter_y_hat + hh2);
	dis2[3] = fabs(pcenter_x_hat - hw2);
	for (int i=0; i < 4; i++)
	{
        //cout<<"line1["<<i<<"].crossnum="<<line1[i].crossnum<<endl;
		if (line1[i].crossnum != 0)
            //cout<<"line1 "<<dis1[i]<<" "<<line1[i].length<<endl;
			area[0] += dis1[i] * line1[i].length;
		if (line2[i].crossnum != 0)
            //cout<<"line2 "<<dis2[i]<<" "<<line2[i].length<<endl;
			area[0] += dis2[i] * line2[i].length;
	}
	area[0] /= 2;
    //cout<<area[0]<<endl;
    area[0] = area[0] / (width1 * height1 + width2 * height2 - area[0]);
}

void Decode(double *preds, double *priors, int n)
{
	for(int i = 0; i < n; i ++)
	{
		double prior_center_x = priors[5*i];
		double prior_center_y = priors[5*i+1];
		double prior_width = priors[5*i+2];
		double prior_height = priors[5*i+3];
		double prior_angle = priors[5*i+4];
		double rbox_center_x = preds[5*i];
		double rbox_center_y = preds[5*i+1];
		double rbox_width = preds[5*i+2];
		double rbox_height = preds[5*i+3];
		double rbox_angle = preds[5*i+4];
		preds[5*i] = rbox_center_x * prior_width + prior_center_x;
		preds[5*i+1] = rbox_center_y * prior_height + prior_center_y;
		preds[5*i+2] = exp(rbox_width) * prior_width;
		preds[5*i+3] = exp(rbox_height) * prior_height;
		if (rbox_angle > 1)  rbox_angle = 1;
		if (rbox_angle < -1) rbox_angle = -1;
		preds[5*i+4] = asin(rbox_angle) * 180 / 3.141593 + prior_angle;
	}
}

struct node
{
	double value;
	int index;
};

bool cmp(struct node a, struct node b)
{
	if(a.value > b.value) return true;
	else return false;
}

void NMS_sub(double *preds, int *indices, double *scores, int *n, double threshold)
{
	int count=0;
	node *scorenodes = new node[n[0]];
	for(int i = 0; i < n[0]; i++)
	{
		scorenodes[i].index = i;
		scorenodes[i].value = scores[i];
	}
	sort(scorenodes, scorenodes + n[0], cmp);
	for(int i = 0; i < n[0]; i++)
	{
		//cout<<i<<" indices[0]="<<indices[0]<<endl;
		int ind_n = scorenodes[i].index;
		bool keep = true;
		for(int j = 0; j < count; j++)
		{
			int ind_p = indices[j];
			double area[1];
			
			//cout<<"Pred:"<<count<<" "<<ind_p<<" "<<preds[5*ind_p]<<" "<<preds[5*ind_p+1]<<" "<<
				//preds[5*ind_p+2]<<" "<<preds[5*ind_p+3]<<" "<<preds[5*ind_p+4]<<endl;
			OverlapSub(preds + ind_p * 5, preds + ind_n * 5, &area[0]);
			//cout<<"AREA="<<area[0]<<endl;
			if (area[0] > threshold)
			{
				keep = false;
				break;
			}
		}
		if(keep)
		{
			indices[count] = ind_n;
			//cout<<i<<" indices[0]=="<<indices[0]<<endl;
			//scores[count] = scorenodes[ind_n].value;
			count ++;
		}
		//cout<<i<<" indices[0]==="<<indices[0]<<endl;
	}
	n[0] = count;
	delete []scorenodes;
}

void NMS_sub_airplane(double *preds, int *indices, double *scores, int *n, double threshold)
{
	int count=0;
	node *scorenodes = new node[n[0]];
	for(int i = 0; i < n[0]; i++)
	{
		scorenodes[i].index = i;
		scorenodes[i].value = scores[i];
	}
	sort(scorenodes, scorenodes + n[0], cmp);
	for(int i = 0; i < n[0]; i++)
	{
		//cout<<i<<" indices[0]="<<indices[0]<<endl;
		int ind_n = scorenodes[i].index;
		bool keep = true;
		for(int j = 0; j < count; j++)
		{
			int ind_p = indices[j];
			double area[1];
			
			//cout<<"Pred:"<<count<<" "<<ind_p<<" "<<preds[5*ind_p]<<" "<<preds[5*ind_p+1]<<" "<<
				//preds[5*ind_p+2]<<" "<<preds[5*ind_p+3]<<" "<<preds[5*ind_p+4]<<endl;
			OverlapSub(preds + ind_p * 5, preds + ind_n * 5, &area[0]);
			//cout<<"AREA="<<area[0]<<endl;
			if (area[0] > threshold)
			{
				keep = false;
				break;
			}
			if (area[0] > 0)
			{
				double xp = preds[ind_p * 5];
				double yp = preds[ind_p * 5 + 1];
				double ap = preds[ind_p * 5 + 4] * 3.1415926 / 180;
				double xn = preds[ind_n * 5];
				double yn = preds[ind_n * 5 + 1];
				double an = preds[ind_n * 5 + 4] * 3.1415926 / 180;
				double ad = atan((yn - yp) / (xn - xp));
				if(fabs(tan(ap - ad)) < 0.2 || fabs(tan(an - ap)) < 0.2)
				{	
					keep = false;
					break;
				} 				
			}
		}
		if(keep)
		{
			indices[count] = ind_n;
			//cout<<i<<" indices[0]=="<<indices[0]<<endl;
			//scores[count] = scorenodes[ind_n].value;
			count ++;
		}
		//cout<<i<<" indices[0]==="<<indices[0]<<endl;
	}
	n[0] = count;
	delete []scorenodes;
}

void NMS_sub_ship(double *preds, int *indices, double *scores, int *n, double threshold)
{
	int count=0;
	node *scorenodes = new node[n[0]];
	for(int i = 0; i < n[0]; i++)
	{
		scorenodes[i].index = i;
		scorenodes[i].value = scores[i];
	}
	sort(scorenodes, scorenodes + n[0], cmp);
	for(int i = 0; i < n[0]; i++)
	{
		//cout<<i<<" indices[0]="<<indices[0]<<endl;
		int ind_n = scorenodes[i].index;
		bool keep = true;
		for(int j = 0; j < count; j++)
		{
			int ind_p = indices[j];
			double area[1];
			
			//cout<<"Pred:"<<count<<" "<<ind_p<<" "<<preds[5*ind_p]<<" "<<preds[5*ind_p+1]<<" "<<
				//preds[5*ind_p+2]<<" "<<preds[5*ind_p+3]<<" "<<preds[5*ind_p+4]<<endl;
			OverlapSub(preds + ind_p * 5, preds + ind_n * 5, &area[0]);
			//cout<<"AREA="<<area[0]<<endl;
			if (area[0] > threshold)
			{
				keep = false;
				break;
			}
		}
		if(keep)
		{
			indices[count] = ind_n;
			//cout<<i<<" indices[0]=="<<indices[0]<<endl;
			//scores[count] = scorenodes[ind_n].value;
			count ++;
		}
		//cout<<i<<" indices[0]==="<<indices[0]<<endl;
	}
	double *arealist = new double[n[0]]();
	n[0] = count;
	count = 0;
	for (int i = 1; i < n[0]; i++)
	{
		int ind_n = indices[i];
		double area[1];
		for(int j = 0; j < i - 1; j++)
		{
			int ind_p = indices[j];
			OverlapSub(preds + ind_p * 5, preds + ind_n * 5, &area[0]);
			arealist[ind_n] += area[0];
			arealist[ind_p] += area[0];
		}
	}
	for (int i = 0; i < n[0]; i++)
	{
		int ind = indices[i];
		if (arealist[ind] > 0.2) continue;
		indices[count] = ind;
		count ++;
	}
	n[0] = count;
	delete []scorenodes;
	delete []arealist;
}

extern "C"
{
	#include <stdio.h>
	double Overlap(double *rbox1, double *rbox2)
	{
		double area[1];
		int i;
		OverlapSub(rbox1, rbox2, area);
		return area[0];
	}
	
	void DecodeAndNMS(double *preds, double *priors, int *indices, double *scores, int *n, double threshold)
	{
		int i;
		Decode(preds, priors, n[0]);
		//for(i=0;i<n[0];i++)
		//{
		//	printf("%f %f %f %f %f\n",preds[5*i],preds[5*i+1],preds[5*i+2],preds[5*i+3],preds[5*i+4]);
		//}
		NMS_sub(preds,indices,scores,n,threshold);
		//for(i=0;i<n[0];i++)
			//printf("%d %f\n",indices[i],scores[i]);
	}
	
	void NMS(double *preds, int *indices, double *scores, int *n, double threshold)
	{
		int i;
		//for(i=0;i<n[0];i++)
		//{
		//	printf("%f %f %f %f %f\n",preds[5*i],preds[5*i+1],preds[5*i+2],preds[5*i+3],preds[5*i+4]);
		//}
		NMS_sub(preds,indices,scores,n,threshold);
	}
	
	void NMS_airplane(double *preds, int *indices, double *scores, int *n, double threshold)
	{
		
		//for(i=0;i<n[0];i++)
		//{
		//	printf("%f %f %f %f %f\n",preds[5*i],preds[5*i+1],preds[5*i+2],preds[5*i+3],preds[5*i+4]);
		//}
		NMS_sub_airplane(preds,indices,scores,n,threshold);
	}
	
	void NMS_ship(double *preds, int *indices, double *scores, int *n, double threshold)
	{
		
		//for(i=0;i<n[0];i++)
		//{
		//	printf("%f %f %f %f %f\n",preds[5*i],preds[5*i+1],preds[5*i+2],preds[5*i+3],preds[5*i+4]);
		//}
		NMS_sub_ship(preds,indices,scores,n,threshold);
	}
	
	void Sum(double *x)
	{
		x[0]=10.0;
	}
	//double DecodeAndNMS
	
}
