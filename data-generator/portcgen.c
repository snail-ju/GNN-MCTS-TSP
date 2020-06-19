#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>

#define MAXN 1000000
#define MAXCOORD 1000000
#define PRANDMAX 1000000000
#define CLUSTERFACTOR 100
#define SCALEFACTOR 1.0

int center[MAXN + 1][2];
int a, b;
int arr[55];
void sprand(int);
double lprand(void);
double normal(void);

int main(int argc, char **argv)
{

	int graph_count;
	int id;
	graph_count = 1000; // number of graph instances to generate

	for (id = 1; id <= graph_count; id++) {
		int N;
		int c;
		int i, j;
		int x, y;
		int seed;
		int nbase;
		double scale;

		N = 20; // number of vertices in the graph instance
		seed = id + 40000;  // random seed

		/* initialize random number generator */
		sprand(seed);

		nbase = N / CLUSTERFACTOR;
		scale = SCALEFACTOR / sqrt((double)N);

		for (i = 1; i <= nbase; i++)
			for (j = 0; j <= 1; j++)
				center[i][j] = (int)(lprand() / PRANDMAX * MAXCOORD);

		char filename[128], tempfp[128];
		FILE *fp = NULL;
		strcpy(filename, "tsp20_c_train.txt");  // storage path of the graph instances
		fp = fopen(filename, "a+");
		printf("%s", filename);

		for (i = 1; i <= N; i++) {
			char coordinate[128];
			c = (int)(lprand() / PRANDMAX * nbase) + 1;
			x = center[c][0] + (int)(normal()*scale*MAXCOORD);
			y = center[c][1] + (int)(normal()*scale*MAXCOORD);
			if (i < N)
				sprintf(coordinate,"%f %f ", 1.0 * x / MAXCOORD, 1.0 * y / MAXCOORD);
			else
				sprintf(coordinate,"%f %f\n", 1.0 * x / MAXCOORD, 1.0 * y / MAXCOORD);
			fprintf(fp, coordinate);
		}
		fclose(fp);
	}
	return 0;
}

double normal(void)	/* Algorithm 3.4.1.P, p. 117, Knuth v. 2 */
{
	static int	goodstill = 0;
	static double	nextvar;
	double		s, t, v1, v2;

	if (goodstill) {
		goodstill = 0;
		return nextvar;
	}
	else {
		goodstill = 1;
		do {
			v1 = 2 * lprand() / PRANDMAX - 1.0;
			v2 = 2 * lprand() / PRANDMAX - 1.0;
			s = v1 * v1 + v2 * v2;
		} while (s >= 1.0);
		t = sqrt((-2.0 * log(s)) / s);
		nextvar = v1 * t;	/* Knuth's x1 */
		return v2 * t;		/* Knuth's x2 */
	}
}

void sprand(seed)
int seed;
{	
	int i, ii;
	int last, next;
	arr[0] = last = seed;
	next = 1;
	for (i = 1; i<55; i++) {
		ii = (21 * i) % 55;
		arr[ii] = next;
		next = last - next;
		if (next < 0) next += PRANDMAX;
		last = arr[ii];
	}
	a = 0;
	b = 24;
	for (i = 0; i<165; i++) last = lprand();
}


double lprand(void)
{
	long t;
	if (a-- == 0) a = 54;
	if (b-- == 0) b = 54;
	t = arr[a] - arr[b];
	if (t < 0) t += PRANDMAX;
	arr[a] = t;
	return (double)t;
}

